import argparse
import copy
import os
from pathlib import Path

import glob
import einops
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid

from cldm.model import create_model, load_state_dict


ckpt = './models/control_sd15_canny.pth'
# ckpt = './models/control_sd15_ini.ckpt'
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(ckpt, location='cuda'))
model = model.cuda()

@torch.no_grad()
def get_curr_reconstruction(curr_latent):
    curr_reconstruction = model.decode_first_stage(curr_latent)
    curr_reconstruction = torch.clamp((curr_reconstruction + 1.0) / 2.0, min=0.0, max=1.0)

    return curr_reconstruction

@torch.no_grad()
def plot_reconstructed_image(curr_latent, fg_image, bg_image, mask):
    curr_reconstruction = get_curr_reconstruction(curr_latent=curr_latent)
    curr_reconstruction = curr_reconstruction[0].cpu().numpy().transpose(1, 2, 0)

    fg_image = torch.clamp((fg_image + 1.0) / 2.0, min=0.0, max=1.0)
    fg_image = fg_image[0].cpu().numpy().transpose(1, 2, 0)

    bg_image = torch.clamp((bg_image + 1.0) / 2.0, min=0.0, max=1.0)
    bg_image = bg_image[0].cpu().numpy().transpose(1, 2, 0)

    mask = mask[0].detach().cpu().numpy().transpose(1, 2, 0)
    composed = fg_image * mask + bg_image * (1 - mask)

    plt.imshow(np.hstack([bg_image, fg_image, composed, curr_reconstruction]))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def opt_loss(
    fg_image: torch.Tensor,
    bg_image: torch.Tensor,
    curr_latent: torch.Tensor,
    mask: torch.Tensor,
    preservation_ratio: float = 100,
):
    # curr_latent.requires_grad_(True)
    curr_reconstruction = model.decode_first_stage(curr_latent)
    loss = (
        F.mse_loss(fg_image * mask, curr_reconstruction * mask)
        + F.mse_loss(bg_image * (1 - mask), curr_reconstruction * (1 - mask))
        * preservation_ratio
    )
    # loss = self.lpips_model(fg_image * mask, curr_reconstruction * mask).sum() + \
    #     self.lpips_model(bg_image * (1 - mask), curr_reconstruction * (1 - mask)).sum()

    return loss

def reconstruct_image_by_optimization(
    fg_image: torch.Tensor, bg_image: torch.Tensor, mask: torch.Tensor, optimization_steps, verbose
):
    encoder_posterior = model.encode_first_stage(fg_image)
    initial_latent = model.get_first_stage_encoding(encoder_posterior)

    
    curr_latent = initial_latent.clone().detach()
    decoder_copy = copy.deepcopy(model.first_stage_model.decoder)
    model.first_stage_model.decoder.requires_grad_(True)
    optimizer = optim.Adam(model.first_stage_model.decoder.parameters(), lr=0.0001)


    for i in tqdm(range(optimization_steps), desc="Reconstruction optimization"):
        if verbose and i % 25 == 0:
            plot_reconstructed_image(
                curr_latent=curr_latent,
                fg_image=fg_image,
                bg_image=bg_image,
                mask=mask,
            )
        optimizer.zero_grad()

        loss = opt_loss(
            fg_image=fg_image, bg_image=bg_image, curr_latent=curr_latent, mask=mask
        )

        if verbose:
            print(f"Iteration {i}: Curr loss is {loss}")

        loss.backward()
        optimizer.step()

    reconstructed_result = model.decode_first_stage(curr_latent)
    
    model.first_stage_model.decoder = None
    model.first_stage_model.decoder = decoder_copy

    return reconstructed_result

def reconstruct_background(samples, init_image, org_mask, optimization_steps):
    reconstructed_samples = []

    for sample in samples:
        optimized_sample = reconstruct_image_by_optimization(
            fg_image=sample.to('cuda').unsqueeze(0),
            bg_image=init_image,
            mask=org_mask,
            optimization_steps=optimization_steps,
            verbose=False
        )
        optimized_sample = (einops.rearrange(optimized_sample, 'b c h w -> b h w c') * 127.5 + 127.5).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
        reconstructed_samples.append(optimized_sample[0])

    # reconstructed_samples = torch.cat(reconstructed_samples)
    return reconstructed_samples

class ImagesDataset(Dataset):
    def __init__(self, img_list):
        self.img_names = img_list

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = Image.open(self.img_names[idx]).convert("RGB")
        image = np.array(image)
        tensor_image = torch.from_numpy((image.astype(np.float32) / 127.5) - 1.0).float()
        tensor_image = einops.rearrange(tensor_image, 'h w c -> c h w').clone()

        return tensor_image

H, W = 512, 512
img_size = (W, H)
mask_size = (W//8, H//8)

# input_folder = '../../BlendingResult/Originals'
# mask_folder = '../../BlendingResult/Masks'
# target_folder = '../../BlendingResult/Default'
# output_folder = '../../BlendingResult/Default_refined'
# os.makedirs(output_folder, exist_ok=True)

# inputfile_list = glob.glob(os.path.join(input_folder, '*.png'))
# for i, inputfile in enumerate(inputfile_list):
#     basename = os.path.basename(inputfile).split('.')[0]
#     print(str(i)+'/'+str(len(inputfile_list)) + ': ' + basename)

#     init_image = Image.open(inputfile).convert("RGB")
#     init_image = np.array(init_image)
#     init_image = torch.from_numpy((init_image.astype(np.float32) / 127.5) - 1.0).float().unsqueeze(0).cuda()
#     init_image = einops.rearrange(init_image, 'b h w c -> b c h w').clone()

#     # mask_path = os.path.join(input_folder, basename + '_mask.png')
#     maskfile_list = glob.glob(os.path.join(mask_folder, f'{basename}_*.png'))
#     for mask_path in maskfile_list:
#         filename = os.path.basename(mask_path).split('.')[0]
#         print(filename)

#         mask = Image.open(mask_path).convert("L")
#         mask = np.array(mask).astype(np.float32) / 255.0
#         mask = mask[None, None]
#         mask[mask < 0.5] = 0
#         mask[mask >= 0.5] = 1
#         mask = torch.from_numpy(mask).to('cuda')

#         filename_list = glob.glob(os.path.join(target_folder, f'{filename}.png'))
#         reconst_img_paths = filename_list

#         samples_dataset = ImagesDataset(
#             img_list=reconst_img_paths,
#         )

#         optimization_steps = 75
#         # returns samples(numpy array) in dataset in list
#         reconstructed_samples = reconstruct_background(samples_dataset, init_image, mask, optimization_steps)

#         for f, res in zip(filename_list, reconstructed_samples):
#             res = Image.fromarray(res)
#             res.save(output_folder + '/' + f'{filename}.png')
# print('Done')

#######################################################################################
input_folder = '../../BlendingResult/for_refinement_2'
output_folder = '../../BlendingResult/for_refinement_2/results'
os.makedirs(output_folder, exist_ok=True)

inputfile_list = glob.glob(os.path.join(input_folder, '*_ori.png'))
for i, inputfile in enumerate(inputfile_list):
    basename = os.path.basename(inputfile).split('_')[0]
    print(str(i)+'/'+str(len(inputfile_list)) + ': ' + basename)

    init_image = Image.open(inputfile).convert("RGB")
    init_image = np.array(init_image)
    init_image = torch.from_numpy((init_image.astype(np.float32) / 127.5) - 1.0).float().unsqueeze(0).cuda()
    init_image = einops.rearrange(init_image, 'b h w c -> b c h w').clone()

    mask_path = os.path.join(input_folder, basename + '_mask.png')
    maskfile_list = [mask_path] #only one mask
    # maskfile_list = glob.glob(os.path.join(mask_folder, f'{basename}_*.png'))
    for mask_path in maskfile_list:
        filename = os.path.basename(mask_path).split('.')[0]
        print(filename)

        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask).to('cuda')

        filename_list = glob.glob(os.path.join(input_folder, f'{basename}_*.png'))
        reconst_img_paths = filename_list

        samples_dataset = ImagesDataset(
            img_list=reconst_img_paths,
        )

        optimization_steps = 75
        # returns samples(numpy array) in dataset in list
        reconstructed_samples = reconstruct_background(samples_dataset, init_image, mask, optimization_steps)

        for f, res in zip(filename_list, reconstructed_samples):
            res = Image.fromarray(res)
            res.save(output_folder + '/' + os.path.basename(f))
print('Done')

######################################################################################


# init_image_path = os.path.join(input_folder, 'ori1.png')
# init_image = Image.open(init_image_path).convert("RGB")
# init_image = np.array(init_image)
# init_image = torch.from_numpy((init_image.astype(np.float32) / 127.5) - 1.0).float().unsqueeze(0).cuda()
# init_image = einops.rearrange(init_image, 'b h w c -> b c h w').clone()

# mask_all_path = os.path.join(input_folder, 'mask1_all.png')
# mask_bot_path = os.path.join(input_folder, 'mask1_bottom.png')

# mask = Image.open(mask_bot_path).convert("L")
# mask = np.array(mask).astype(np.float32) / 255.0
# mask = mask[None, None]
# mask[mask < 0.5] = 0
# mask[mask >= 0.5] = 1
# mask = torch.from_numpy(mask).to('cuda')

# filename_list = glob.glob(os.path.join(input_folder, 'ori1_content.png'))
# # reconst_img_paths = ['./inputs/unreal_edit.png']
# reconst_img_paths = filename_list

# samples_dataset = ImagesDataset(
#     img_list=reconst_img_paths,
# )

# optimization_steps = 75
# # returns samples(numpy array) in dataset in list
# reconstructed_samples = reconstruct_background(samples_dataset, init_image, mask, optimization_steps)

# for f, res in zip(filename_list, reconstructed_samples):
#     res = Image.fromarray(res)
#     res.save(output_folder + '/' + os.path.basename(f))