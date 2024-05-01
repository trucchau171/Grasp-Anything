import os
import functools
import os.path as osp
import math
import argparse
import time
import numpy as np
from tqdm import tqdm


import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

# from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from diffusion.resample import create_named_schedule_sampler
from diffusion.fp16_util import MixedPrecisionTrainer

from utils.model_util import create_gaussian_diffusion, get_default_diffusion

"""
Running sample:
python train_contactformer.py --train_data_dir ../data/proxd_train --valid_data_dir ../data/proxd_valid --fix_ori --epochs 1000 --jump_step 8
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Main Detection Training')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--is-aligned', type=int, default=0,
                        help='Whether to perform alignment of text & seg map')

    # Datasets
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--seen', type=int, default=1,
                        help='Flag for using seen classes, only work for Grasp-Anything dataset') 

    args = parser.parse_args()
    return args


def validate(net, device, val_data, iou_threshold, diffusion):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    use_ddim = False  # FIXME - hardcoded
    clip_denoised = False  # FIXME - hardcoded

    ld = len(val_data)

    with torch.no_grad():
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        for x, y, didx, rot, zoom_factor, prompt in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]

            sample = sample_fn(
                net,
                yc.shape,
                given_x=xc,
                clip_denoised=clip_denoised,
                model_kwargs=None,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
                # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
            )

            y_pos, y_cos, y_sin, y_width = yc
            [pos_pred, cos_pred, sin_pred, width_pred] = sample['pred_xstart']
            pred_sample = sample['sample']

            recon_loss_semantics = ((yc-pred_sample)**2).mean()

            p_loss = F.smooth_l1_loss(pos_pred, y_pos)
            cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
            sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
            width_loss = F.smooth_l1_loss(width_pred, y_width)


            # lossd = net.compute_loss(xc, yc, prompt)
            lossd =  {
                'loss': p_loss + cos_loss + sin_loss + width_loss,
                'losses': {
                    'p_loss': p_loss,
                    'cos_loss': cos_loss,
                    'sin_loss': sin_loss,
                    'width_loss': width_loss
                }
            }

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, rot.item(), zoom_factor.item()),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results, recon_loss_semantics


def train(epoch, net, diffusion, device, train_data, optim, batches_per_epoch, lr=1e-3, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """


    # Create diffusion sampler, optimizer, and trainer
    schedule_sampler_type = 'uniform'
    schedule_sampler = create_named_schedule_sampler(schedule_sampler_type, diffusion)

    use_fp16 = False
    fp16_scale_growth = 1e-3
    mp_trainer = MixedPrecisionTrainer(
            model=net,
            use_fp16=use_fp16,
            fp16_scale_growth=fp16_scale_growth,
    )

    if optim == 'adam':
        optimizer = AdamW(
        mp_trainer.master_params, lr=lr
    )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(optim))


    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()
    torch.autograd.set_detect_anomaly(True)

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for x, y, _, _, _, prompt in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            t, weights = schedule_sampler.sample(y.shape[0], device)

            # Initialize the optimizer's step
            mp_trainer.zero_grad()

            # x shape [8, 3, 224, 224]

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            # lossd = net.compute_loss(xc, yc, prompt)
            # lossd = net.compute_loss(xc, yc)

            lossd = diffusion.training_losses(net, yc, t, xc)

            loss = (lossd['loss'] * weights).mean()


            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    device = 'cuda'

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      seen=args.seen)
    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size,
        # is_aligned=args.is_aligned
    )

    net = net.to(device)
    diffusion = create_gaussian_diffusion(get_default_diffusion())
    logging.info('Done')

    # if args.optim.lower() == 'adam':
    #     optimizer = optim.Adam(net.parameters())
    # elif args.optim.lower() == 'sgd':
    #     optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # else:
    #     raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # Print model architecture.
    # summary(net, (input_channels, args.input_size, args.input_size))
    # f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    # sys.stdout = f
    # summary(net, (input_channels, args.input_size, args.input_size))
    # sys.stdout = sys.__stdout__
    # f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, diffusion, device, train_data, args.optim.lower(), args.batches_per_epoch, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results, recon_loss_semantics = validate(net, device, val_data, args.iou_threshold, diffusion)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        logging.info('recon_loss_semantics Loss: {:0.4f}'.format(recon_loss_semantics))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()




# if __name__ == '__main__':
#     # torch.manual_seed(0)
#     print(torch.version.cuda)
#     # torch.multiprocessing.set_start_method('spawn')
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument("--train_data_dir", type=str, default="data/proxd_train",
#                         help="path to POSA_temp dataset dir")
#     parser.add_argument("--valid_data_dir", type=str, default="data/proxd_valid",
#                         help="path to POSA_temp dataset dir")
#     parser.add_argument("--load_ckpt", type=str, default=None,
#                         help="load a checkpoint as the continue point for training")
#     parser.add_argument("--posa_path", type=str, default="training/posa/model_ckpt/best_model_recon_acc.pt")
#     parser.add_argument("--out_dir", type=str, default="training/", help="Folder that stores checkpoints and training logs")
#     parser.add_argument("--experiment", type=str, default="default_experiment",
#                         help="Experiment name. Checkpoints and training logs will be saved in out_dir/experiment folder.")
#     parser.add_argument("--save_interval", type=int, default=50, help="Epoch interval for saving model checkpoints.")
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--epochs", type=int, default=1000)
#     parser.add_argument('--fix_ori', dest='fix_ori', action='store_const', const=True, default=False,
#                         help="fix orientation of each segment with the rotation calculated from the first frame")
#     parser.add_argument("--encoder_mode", type=int, default=1,
#                         help="Encoder mode (different number represents different variants of encoder)")
#     parser.add_argument("--decoder_mode", type=int, default=1,
#                         help="Decoder mode (different number represents different variants of decoder)")
#     parser.add_argument("--n_layer", type=int, default=3, help="number of layers in transformer")
#     parser.add_argument("--n_head", type=int, default=4, help="number of heads in transformer")
#     parser.add_argument("--dim_ff", type=int, default=512, help="dimension of hidden layers in positionwise MLP in the transformer")
#     parser.add_argument("--f_vert", type=int, default=64, help="dimension of the embeddings for body vertices")
#     parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
#     parser.add_argument("--jump_step", type=int, default=8, help="frame skip size for each input motion sequence")
#     parser.add_argument("--max_frame", type=int, default=256, help="The maximum length of motion sequence (after frame skipping) which model accepts.")
#     parser.add_argument("--eval_epochs", type=int, default=10, help="The number of epochs that we periodically evalute the model.")
#     parser.add_argument("--datatype", type=str, default="proxd", help="Dataset type indicator: PRO-teXt or HUMANISE.")

#     args = parser.parse_args()
#     args_dict = vars(args)

#     # Parse arguments
#     train_data_dir = args_dict['train_data_dir']
#     valid_data_dir = args_dict['valid_data_dir']
#     load_ckpt = args_dict['load_ckpt']
#     save_interval = args_dict['save_interval']
#     out_dir = args_dict['out_dir']
#     experiment = args_dict['experiment']
#     lr = args_dict['lr']
#     epochs = args_dict['epochs']
#     fix_ori = args_dict['fix_ori']
#     encoder_mode = args_dict['encoder_mode']
#     decoder_mode = args_dict['decoder_mode']
#     n_layer = args_dict['n_layer']
#     n_head = args_dict['n_head']
#     num_workers = args_dict['num_workers']
#     jump_step = args_dict['jump_step']
#     max_frame = args_dict['max_frame']
#     dim_ff = args_dict['dim_ff']
#     f_vert = args_dict['f_vert']
#     posa_path = args_dict['posa_path']
#     eval_epochs = args_dict['eval_epochs']
#     datatype = args_dict['datatype']

#     save_ckpt_dir = os.path.join(out_dir, experiment, "model_ckpt")
#     log_dir = os.path.join(out_dir, experiment, "tb_log")
#     os.makedirs(save_ckpt_dir, exist_ok=True)
#     device = torch.device("cuda")
#     dtype = torch.float32
#     kl_w = 0.5

#     if datatype == "proxd":
#         train_dataset = ProxDataset_txt(train_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
#                                     step_multiplier=1, jump_step=jump_step)
#         train_data_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=num_workers)
#         valid_dataset = ProxDataset_txt(valid_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
#                                     step_multiplier=1, jump_step=jump_step)
#         valid_data_loader = DataLoader(valid_dataset, batch_size=6, shuffle=True, num_workers=num_workers)
#     else:
#         train_dataset = HUMANISE(train_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
#                                     step_multiplier=1, jump_step=jump_step)
#         train_data_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=num_workers)
#         valid_dataset = HUMANISE(valid_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
#                                     step_multiplier=1, jump_step=jump_step)
#         valid_data_loader = DataLoader(valid_dataset, batch_size=6, shuffle=True, num_workers=num_workers)
    
#     # Create model and diffusion object
#     model, diffusion = create_model_and_diffusion(datatype)
#     print(
#         f"Training using model----encoder_mode: {encoder_mode}, decoder_mode: {decoder_mode}, max_frame: {max_frame}, "
#         f"using_data: {train_data_dir}, epochs: {epochs}, "
#         f"n_layer: {n_layer}, n_head: {n_head}, f_vert: {f_vert}, dim_ff: {dim_ff}, jump_step: {jump_step}")
#     print("Total trainable parameters: {}".format(count_parameters(model)))
#     # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=0.0001, patience=10, verbose=True)
#     # scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000, gamma=0.1, verbose=True)
#     # milestones = [200, 400, 600, 800]
#     # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, verbose=True)

#     best_valid_loss = float('inf')
#     best_cfd= float('inf')

#     starting_epoch = 0
#     if load_ckpt is not None:
#         checkpoint = torch.load(load_ckpt)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         starting_epoch = checkpoint['epoch'] + 1
#         print('loading stats of epoch {}'.format(starting_epoch))


#     writer = SummaryWriter(log_dir)

#     for e in range(starting_epoch, epochs):
#         print('Training epoch {}'.format(e))
#         start = time.time()
#         total_train_loss = train()
#         training_time = time.time() - start
#         print('training_time = {:.4f}'.format(training_time))

#         if e % save_interval == save_interval-1:
#             start = time.time()
#             total_valid_loss, total_cfd, total_acc = validate()
#             validation_time = time.time() - start
#             print('validation_time = {:.4f}'.format(validation_time))

#             data = {
#                 'epoch': e,
#                 'model_state_dict': model.state_dict(),
#                 # 'optimizer_state_dict': optimizer.state_dict(),
#                 'total_train_loss': total_train_loss,
#                 'total_valid_loss': total_valid_loss,
#             }
#             torch.save(data, osp.join(save_ckpt_dir, 'epoch_{:04d}.pt'.format(e)))

#             if total_valid_loss < best_valid_loss:
#                 print("Updated best model due to new lowest valid_loss. Current epoch: {}".format(e))
#                 best_valid_loss = total_valid_loss
#                 data = {
#                     'epoch': e,
#                     'model_state_dict': model.state_dict(),
#                     'total_train_loss': total_train_loss,
#                     'total_valid_loss': total_valid_loss,
#                 }
#                 torch.save(data, osp.join(save_ckpt_dir, 'best_model_valid_loss.pt'))

#             if total_cfd < best_cfd:
#                 print("Updated best model due to new highest total_cfd. Current epoch: {}".format(e))
#                 best_cfd = total_cfd
#                 data = {
#                     'epoch': e,
#                     'model_state_dict': model.state_dict(),
#                     'total_train_loss': total_train_loss,
#                     'total_valid_loss': total_valid_loss,
#                     'total_cfd': total_cfd
#                 }
#                 torch.save(data, osp.join(save_ckpt_dir, 'best_model_cfd.pt'))
