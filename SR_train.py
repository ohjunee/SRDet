import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.model import SR_Model
import SR_test
from models.experimental import attempt_load
from utils.SRdataset import create_SRdataloader
from utils.general import increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, set_logging, one_cycle, colorstr
from utils.loss import SR_Loss
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.SR_utility import quantize, calc_psnr

logger = logging.getLogger(__name__)

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    # Model
    pretrained = weights.endswith('.pt')
    model = SR_Model(opt.cfg, ch=3).to(device)


    # Freeze
    freeze = []  # parameter names to freeze (full or partial)  'model.%s.' % x for x in range(8)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size= hyp['lr_decay'], gamma=hyp['gamma'])

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    # DIV2k dataset
    dataloader, dataset = create_SRdataloader(opt, train=True, batch_size=opt.batch_size, rank=rank, world_size=opt.world_size, workers= opt.workers)
    nb = len(dataloader)
    scaler = amp.GradScaler(enabled=cuda)
    scheduler.last_epoch = start_epoch - 1  # do not move
    sr_loss = SR_Loss(opt, device)
    testloader, _ = create_SRdataloader(opt, train=False, batch_size=1, rank=rank, world_size=opt.world_size, workers= opt.workers)


    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(1, device=device)
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'loss', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (lr, hr, _) in pbar:
            ni = i + nb * epoch # number integrated batches (since train start)
            idx_scale = opt.scale
            lr = lr.to(device).float()
            hr = hr.to(device).float()

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(lr)  # forward
                loss = sr_loss(pred, hr)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 2) % ('%g/%g' % (epoch, epochs - 1), mem, mloss, lr.shape[-1])
                pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        learning_rate = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # PSNR
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            model.eval()
            with torch.no_grad():
                for idx_scale, scale in enumerate(opt.scale):
                    eval_acc = 0
                    #testloader.dataset.set_scale(idx_scale)
                    pbar = enumerate(testloader)
                    pbar = tqdm(pbar, total=len(testloader))
                    for idx_img, (lr, hr, filename) in pbar:
                        lr = lr.to(device).float()
                        hr = hr.to(device).float()
                        filename = filename[0]
                        pred = model(lr, idx_scale)
                        pred = quantize(pred, opt.rgb_range)
                        save_list = [pred]
                        eval_acc += calc_psnr(pred, hr, scale, opt.rgb_range)
                        save_list.extend([lr, hr])
                # PSNR 로그로 표시
                results = eval_acc / len(testloader)
                logger.info(f'[DIV2K x{opt.scale}]\tPSNR: {results}')
            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 1 % (results) + '\n')  # append metrics, val_loss
            # Update best PSNR
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
        # Save model

        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'training_results': results_file.read_text(),
                'model': deepcopy(model.module if is_parallel(model) else model).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict(),
                'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

        # Save last, best and delete
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        if wandb_logger.wandb:
            if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                wandb_logger.log_model(
                    last.parent, opt, epoch, fi, best_model=best_fitness == fi)
        del ckpt

        # (SR)end epoch ----------------------------------------------------------------------------------------------------
    # end training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='EDSR.yaml', help='model.yaml')
    parser.add_argument('--data', type=str, default='data/DIV2K.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.SR_train.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=1000) # detection epoch -> 300, SR epoch -> 1000
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='EDSR', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # SR_model training
    parser.add_argument('--dir_data', type=str, default='../SRDet', help='dataset directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='path to the pre-trained model')
    parser.add_argument('--patch_size', type=int, default=96, help='output patch size')  # 192 # 80 # 96
    parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')
    parser.add_argument('--n_train', type=int, default=800, help='number of training set')
    parser.add_argument('--n_val', type=int, default=100, help='number of validation set')
    parser.add_argument('--scale', default='2', help='super resolution scale')
    parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
    parser.add_argument('--offset_val', type=int, default=800, help='validation index offest')
    parser.add_argument('--noise', type=str, default='.', help='Gaussian noise std.')
    parser.add_argument('--loss', type=str, default='1*L1', help='loss function configuration')
    opt = parser.parse_args()

    opt.scale = list(map(lambda x : int(x), opt.scale.split('+')))
    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.cfg, opt.hyp =  check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device, tb_writer)