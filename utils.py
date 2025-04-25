import argparse

import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F


# Initial configs, some values need updates!
opt = {
    # Required paths
    'dataroot_gt': 'data/FFHQ/train/GT',
    'meta_info': 'data/FFHQ/train/meta_info_FFHQ5000sub_GT.txt',

    'io_backend': {'type': 'disk'},
    
    # Augmentation settings
    'use_hflip': True,
    'use_rot': False,
    
    # First degradation kernel settings
    'blur_kernel_size': 21,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob': 0.1,
    'blur_sigma': [0.2, 3],
    'betag_range': [0.5, 4],
    'betap_range': [1, 2],
    
    # Second degradation kernel settings
    'blur_kernel_size2': 21,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob2': 0.1,
    'blur_sigma2': [0.2, 1.5],
    'betag_range2': [0.5, 4],
    'betap_range2': [1, 2],
    
    # Final sinc filter setting
    'final_sinc_prob': 0.8,
    
    # Image size
    'gt_size': 512,
    'scale': 4,


    'gt_usm': True,  # USM the ground-truth

    # the first degradation process
    'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
    'resize_range': [0.2, 1.5],
    'gaussian_noise_prob': 0.5,
    'noise_range': [1, 20],
    'poisson_scale_range': [0.05, 2],
    'gray_noise_prob': 0.4,
    'jpeg_range': [50, 95],

    # the second degradation process
    'second_blur_prob': 0.8,
    'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
    'resize_range2': [0.3, 1.2],
    'gaussian_noise_prob2': 0.5,
    'noise_range2': [1, 15],
    'poisson_scale_range2': [0.05, 1.5],
    'gray_noise_prob2': 0.4,
    'jpeg_range2': [70, 95],

    'gt_size': 512,
    'queue_size': 176,  # divisible by batch size 8
}

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inferring", 
        action="store_true",
        default=False,
        help="Set to True for inferring, False for training.",
    )

    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device to use for training. Choose from 'cuda' or 'cpu'.",
    )

    # ------------------------ path & save related -------------------------
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        # required=True,
        default="./data",
        help="Directory containing the dataset. It should contain train and val folders.",
    )
    parser.add_argument(
        "--save_model", "-w",
        type=str,
        # required=True,
        default="./model.pth",
        help=".../.../.../model.pth",
    )
    parser.add_argument(
        "--test_input_dir", "-ti",
        type=str,
        default="./data/test/LQ",
        help="Directory containing the test images.",
    )
    parser.add_argument(
        "--test_output_dir", "-to",
        type=str,
        default="./data/test/preds",
        help="Directory to save the test results.",
    )
    # ----------------------------------------------------------------------


    # ----------------------- data loader related --------------------------
    parser.add_argument(
        "--batch_size", "-bs",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num_workers", "-nw",
        type=int,
        default=4,
        help="Number of workers for the data loader.",
    )
    # ----------------------------------------------------------------------


    # ----------------------- training related -----------------------------
    parser.add_argument(
        "--num_epochs", "-ep",
        type=int,
        default=200,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=0.0002,
        help="Learning rate for the optimizer.",
    )
    # ----------------------------------------------------------------------






    return parser

def modify_opt(args, opt):
    # Assuming we're facing two folders:
        # /train and /val
    # Update the paths in the options dictionary first!
    opt['dataroot_gt'] = args.input_dir + '/train/GT'
    opt['meta_info'] = args.input_dir + '/train/meta_info_FFHQ6000sub_GT.txt'

    opt['val_gt'] = args.input_dir + '/val/GT'
    opt['val_lq'] = args.input_dir + '/val/LQ'

    return opt

# Also move them to input device
def degradation(batch_dict, opt, device):
    """Apply degradation to generate LQ images from GT images.
    
    Args:
        batch_dict (dict): Dictionary containing GT images and kernels from dataloader
        opt (dict): Configuration dictionary with degradation parameters
        device (str): Device to process images on
    
    Returns:
        dict: Dictionary containing both GT and LQ images
    """
    # Initialize components for degradation
    jpeger = DiffJPEG(differentiable=False).to(device)
    usm_sharpener = USMSharp().to(device) if opt.get('gt_usm', True) else None
    
    # Move data to device
    gt = batch_dict['gt'].to(device)
    kernel1 = batch_dict['kernel1'].to(device)
    kernel2 = batch_dict['kernel2'].to(device)
    sinc_kernel = batch_dict['sinc_kernel'].to(device)
    
    # USM sharpen the GT images
    if usm_sharpener is not None:
        gt = usm_sharpener(gt)
    
    # Record original size
    ori_h, ori_w = gt.size()[2:4]
    
    # ----------------------- First degradation process ----------------------- #
    # 1. blur
    out = filter2D(gt, kernel1)
    
    # 2. random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    
    # 3. add noise
    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    
    # 4. JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)
    
    # ----------------------- Second degradation process ----------------------- #
    # 5. blur
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)
    
    # 6. random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
    
    # 7. add noise
    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    
    # 8. JPEG compression + final sinc filter (two orders)
    if np.random.uniform() < 0.5:
        # order 1: resize back + sinc filter, then JPEG
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # order 2: JPEG then resize back + sinc filter
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
    
    # 9. clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    
    # 10. random crop
    if opt.get('gt_size', 0) > 0:
        gt_size = opt['gt_size']
        gt, lq = paired_random_crop(gt, lq, gt_size, opt['scale'])
    
    # Return both GT and LQ
    return {'gt': gt, 'lq': lq}

# return a list! do the average in the end
def compute_psnr(sr_batch, gt_batch):
    sr_batch = sr_batch.detach().cpu().numpy()
    gt_batch = gt_batch.detach().cpu().numpy()

    batch_size = sr_batch.shape[0]
    psnr_values = []

    for i in range(batch_size):
        # Get single image from batch
        sr_img = sr_batch[i].transpose(1, 2, 0)  # CHW to HWC
        gt_img = gt_batch[i].transpose(1, 2, 0)  # CHW to HWC
        
        # Scale to 0-255 range as in evaluate.py's read_image function
        sr_img = sr_img * 255.0
        gt_img = gt_img * 255.0
        
        # Use the exact PSNR function from evaluate.py
        mse_value = np.mean((sr_img - gt_img)**2)
        psnr_value = 20. * np.log10(255. / np.sqrt(mse_value))
        psnr_values.append(psnr_value)
    
    assert len(psnr_values) == batch_size, "Mismatch in PSNR values count"
    
    return psnr_values