# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from json import encoder
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Block
from model.architecture.util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_2d_sincos_pos_embed_spectogram, get_1d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from model.architecture.util.misc import concat_all_gather
from model.architecture.util.patch_embed import PatchEmbed_new, PatchEmbed_org, PatchEmbedding2D, PatchEmbedding1D
import torch.fft

# SwinTransformerBlock is only used when decoder_mode=1 (not the default paper model)
try:
    from timm.models.swin_transformer import SwinTransformerBlock
except ImportError:
    SwinTransformerBlock = None

from tslearn.metrics import dtw
from tslearn.metrics import SoftDTWLossPyTorch


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 audio_exp=False, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 use_custom_patch=False, split_pos=False, pos_trainable=False, use_nce=False, beta=4.0, decoder_mode=0,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, no_shift=False,
                 ):
        super().__init__()

        self.audio_exp=audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if use_custom_patch:
            print(f'Use custom patch_emb with patch size: {patch_size}, stride: {stride}')
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        else:
            self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        self.use_custom_patch = use_custom_patch
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding


        self.no_shift=no_shift


        self.decoder_mode = decoder_mode
        if self.use_custom_patch: # overlapped patches as in AST. Similar performance yet compute heavy
            window_size= (6,6)
            feat_size = (102,12)
        else:
            window_size= (4,4)
            feat_size = (64,8)                
        if self.decoder_mode == 1:
            decoder_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0,0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0,0)
                    else:
                        shift_size = (2,0)
                    #shift_size = tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size])
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        feat_size=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        drop_attn=0.0,
                        drop_path=0.0,
                        extra_norm=False,
                        sequential_attn=False,
                        norm_layer=norm_layer, #nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)        
        else:
            # Transfomer
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax=nn.LogSoftmax(dim=-1)

        self.mask_t_prob=mask_t_prob
        self.mask_f_prob=mask_f_prob
        self.mask_2d=mask_2d

        self.epoch = epoch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)    
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.audio_exp:   
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        else:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        if self.audio_exp:
            if self.use_custom_patch: # overlapped patch
                h,w = self.patch_embed.patch_hw
                # todo: fixed h/w patch size and stride size. Make hw custom in the future
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h*w, p**2 * 1))
                #x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                #x = torch.einsum('nchpwq->nhwpqc', x)
                #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                #h,w = self.patch_embed.patch_hw
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum('nchpwq->nhwpqc', x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]    
        h = 1024//p
        w = 128//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch: # overlapped patch
            T=101
            F=12
        else:            
            T=64
            F=8
        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio, mask_2d=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #emb = self.encoder_emb(x)

        return x, mask, ids_restore, None

    def forward_encoder_no_mask(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        #x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        contextual_embs=[]
        for n, blk in enumerate(self.blocks):
            x = blk(x)
            if n > self.contextual_depth:
                contextual_embs.append(self.norm(x))
        #x = self.norm(x)
        contextual_emb = torch.stack(contextual_embs,dim=0).mean(dim=0)

        return contextual_emb

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        if self.decoder_mode != 0:
            B,L,D=x.shape
            x = x[:,1:,:]
            if self.use_custom_patch:
                x = x.reshape(B,101,12,D)
                x = torch.cat([x,x[:,-1,:].unsqueeze(1)],dim=1) # hack
                x = x.reshape(B,1224,D)
        if self.decoder_mode > 3: # mvit
            x = self.decoder_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B,102,12,256)
                pred = pred[:,:101,:,:]
                pred = pred.reshape(B,1212,256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]
        return pred, None, None #emb, emb_pixel

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3], reconstruced 
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss      

    def forward(self, imgs, mask_ratio=0.8):
        emb_enc, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio, mask_2d=self.mask_2d)
        pred, _, _ = self.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]
        loss_recon = self.forward_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)
        loss_contrastive = torch.FloatTensor([0.0]).cuda()
        return loss_recon, pred, mask, loss_contrastive


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

######################################################################################
## adapted from here ##
######################################################################################
from scipy import signal as scipy_signal
from typing import Tuple, Dict, Optional
import numpy as np

class FrequencyBaseline(nn.Module):
    """
    Simpler baseline model for 1D sinusoidal signal reconstruction:
    1. Randomly masks patches of the input signal
    2. Computes main frequency, amplitude, and peak-to-peak from visible patches
    3. Stores these as latent features
    4. Reconstructs masked patches using computed parameters
    5. Computes MSE on masked patches
    """

    def __init__(self, patch_size: int, in_chans: int = 1):
        """
        Args:
            patch_size: Size of each patch
            in_chans: Number of input channels (default 1 for 1D signal)
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.latent_dim = 3  # frequency, amplitude, phase

    def _compute_signal_parameters_batch(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency, amplitude, peak-to-peak, and phase from signals using PyTorch.

        Args:
            signal_tensor: Tensor of shape (B,) or (B, L) containing signal data

        Returns:
            Tensor of shape (B, 3) with [frequency, amplitude, phase]
        """
        device = signal_tensor.device

        if signal_tensor.dim() == 1:
            signal_tensor = signal_tensor.unsqueeze(0)

        B = signal_tensor.shape[0]
        params = torch.zeros(B, self.latent_dim, device=device)

        # Compute amplitude (RMS)
        params[:, 1] = torch.sqrt(torch.mean(signal_tensor ** 2, dim=1))

        # Compute frequency and phase using FFT
        if signal_tensor.shape[1] > 1:
            fft = torch.fft.rfft(signal_tensor, dim=1)
            fft_mag = torch.abs(fft)
            fft_phase = torch.angle(fft)
        else:
            fft_mag = torch.zeros_like(signal_tensor)
            fft_phase = torch.zeros_like(signal_tensor)

        # Find dominant frequency (skip DC component at index 0)
        if fft_mag.shape[1] > 1:
            # Get argmax of frequencies (excluding DC)
            _, freq_idx = torch.max(fft_mag[:, 1:], dim=1)
            freq_idx = freq_idx + 1  # Offset by 1 to account for skipping DC

            # Normalize frequency
            freq_vals = freq_idx.float() / signal_tensor.shape[1]
            params[:, 0] = freq_vals

            # Get phase at dominant frequency
            params[:, 2] = fft_phase[torch.arange(B), freq_idx]

        # Handle zero signals
        zero_mask = torch.sum(torch.abs(signal_tensor), dim=1) < 1e-6
        params[zero_mask] = 0.0

        return params

    def _reconstruct_from_parameters_batch(self, params: torch.Tensor, patch_size: int, device) -> torch.Tensor:
        """
        Reconstruct signal patches using computed parameters (vectorized).

        Args:
            params: Tensor of shape (B, 4) with [frequency, amplitude, peak_to_peak, phase]
            patch_size: Size of each patch
            patch_offsets: Tensor of shape (B,) with time offset for each sample
            device: Torch device

        Returns:
            Reconstructed patches as tensor of shape (B, patch_size)
        """
        B = params.shape[0]

        # Create time indices: (B, patch_size)
        t = torch.arange(patch_size, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1)

        freq = params[:, 0].unsqueeze(1)  # (B, 1)
        amplitude = params[:, 1].unsqueeze(1)  # (B, 1)
        phase = params[:, 2].unsqueeze(1)  # (B, 1)

        # Reconstruct: amplitude * sin(2*pi*freq*t + phase)
        reconstructed = amplitude * torch.sin(2 * np.pi * freq * t + phase)

        return reconstructed

    def patch_embed(self, samples: torch.Tensor):
        """
        Patchify input signal into non-overlapping patches.

        Args:
            samples: Input tensor of shape (B, C, seq_length)

        Returns:
            Patches tensor of shape (B, num_patches, patch_size, C)
        """
        B, C, seq_length = samples.shape
        num_patches = seq_length // self.patch_size

        patches = samples.unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3)
        return patches

    def forward(self, samples: torch.Tensor, mask_ratio: float = 0.65,
                mask: Optional[torch.Tensor] = None,
                ids_restore: Optional[torch.Tensor] = None,
                ids_keep: Optional[torch.Tensor] = None) -> Tuple:
        """
        Forward pass of frequency baseline model.

        Args:
            samples: Input tensor (B, C, seq_length) or (B, seq_length)
            mask_ratio: Ratio of patches to mask
            mask: Optional pre-computed mask
            ids_restore: Optional indices to restore original order

        Returns:
            Tuple containing:
            - recon_loss: MSE loss on masked patches only
            - pred: Reconstructed signal (B, num_patches, patch_size * in_chans)
            - mask: Binary mask (B, num_patches)
            - padding_mask: Valid region mask
            - latent: Latent features (B, num_patches, 4) - [freq, amp, ptp, phase]
            - target: Target patches (B, num_patches, patch_size * in_chans)
        """
        device = samples.device

        # Handle input shape: (B, C, seq_length) or (B, seq_length)
        if samples.dim() == 2:
            samples = samples.unsqueeze(1)  # Add channel dimension

        B, C, seq_length = samples.shape
        assert C == self.in_chans, f"Expected {self.in_chans} channels, got {C}"

        # Padding mask
        padding_mask = self.get_padding_mask(samples)

        num_patches = seq_length // self.patch_size

        # Generate mask if not provided
        if mask is None:
            num_masked = max(1, int(num_patches * mask_ratio))
            masked_indices = np.random.choice(num_patches, num_masked, replace=False)
            mask = torch.zeros(B, num_patches, device=device)
            mask[:, masked_indices] = 1
        else:
            mask = mask.to(device)

        # Patchify input: (B, C, seq_length) -> (B, num_patches, patch_size * C)
        patches = self.patch_embed(samples)

        # Store target (ground truth patches)
        target = patches.clone()
        target = target[:, :, 2, :]

        # Initialize reconstruction and latent features
        pred = torch.zeros(B, num_patches, self.patch_size, device=device)
        latent = torch.zeros(B, self.latent_dim, device=device)

        # Fully vectorized batch processing (same mask across batch)
        # Get indices of masked and visible patches (same for all samples)
        masked_patches_indices = torch.where(mask[0] == 1)[0]  # (num_masked,)
        visible_patches_indices = torch.where(mask[0] == 0)[0]  # (num_visible,)

        # Concatenate all visible patches across all samples in the batch
        # Shape: (B, num_visible, patch_size) - taking only the ppg channel!!!
        visible_patches_all = patches[:, visible_patches_indices, 2, :]

        # Compute parameters from concatenated visible signal (all visible patches, all samples)
        visible_signal_all = visible_patches_all.flatten(1)  # Flatten across patches
        latent = self._compute_signal_parameters_batch(visible_signal_all)  # (1, 3)

        # Reconstruct masked patches using parameters from all visible patches
        if len(masked_patches_indices) > 0:
            # Reconstruct all masked patches at once
            reconstructed_patches = self._reconstruct_from_parameters_batch(
                latent,
                self.patch_size,
                device
            )  # (B, patch_size)

            pred[:, masked_patches_indices, :] = reconstructed_patches.unsqueeze(1).repeat(1, len(masked_patches_indices), 1)

        # Compute MSE loss only on masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # (B, num_patches, patch_size*C)
        recon_loss = torch.mean((pred - target) ** 2 * mask_expanded)

        # make the latent similar to the original model shape: (B, num_patches, latent_dim)
        latent = latent.unsqueeze(1).repeat(1, num_patches+1, 1)  # (B, num_patches, latent_dim)

        return recon_loss, pred, mask, padding_mask, latent, target, None, None, None, None, None

    def forward_encoder(self, x, mask_ratio):
        "the same as forward , but returning all the returned arguments from the forward pass except the None ones"
        recon_loss, pred, mask, padding_mask, latent, target, _, _, _, _, _ = self.forward(x, mask_ratio)
        return target, mask, _, latent, padding_mask

    def forward_loss(self, target, pred, mask, padding_mask):
        # Patchify input: (B, C, seq_length) -> (B, num_patches, C, patch_size)
        target = self.patch_embed(target)
        if target.shape[2] == 3:
            target = target[:, :, 2, :]
        # Compute MSE loss only on masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # (B, num_patches, patch_size*C)
        recon_loss = torch.mean((pred - target) ** 2 * mask_expanded)
        return None, recon_loss, None, None, None, None, None

    def get_latent_dim(self) -> int:
        """Return dimensionality of latent features"""
        return self.latent_dim

    def unpatchify(self, x):
        """
        Reconstruct time-series from patches.
        x: (Batch, num_patches, patch_size * in_chans)
        Returns: (Batch, in_chans, seq_length)
        """
        B, N, D = x.shape
        C = D // self.patch_size
        x = x.view(B, N, C, self.patch_size).permute(0, 2, 1, 3).reshape(B, C,
                                                                         N * self.patch_size)  # (N, in_chans, seq_length)
        return x

    def patchify(self, x):
        """
        Converts time-series input into patches.
        x: (N, in_chans, seq_length)
        Returns: (N, num_patches, patch_size * in_chans)
        """
        N, C, L = x.view(x.size(0), x.size(1), -1).shape
        x = x.view(N, C, L // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(N, L // self.patch_size, -1)  # (N, num_patches, patch_size * in_chans)
        return x

    def get_padding_mask(self, x):
        """
        Identify padded regions in the input sequence, in a same way as previous function but in an unpacthified sequence.
        Args:
            x: Input tensor of shape [B, C, seq_len]
        Returns:
            padding_mask: Binary tensor [B, num_patches], where 1 marks padded patches, for a patchified sample
        """
        B, C, seq_len = x.shape
        patch_size = self.patch_size
        num_patches = seq_len // patch_size
        padding_reference = x[:, :, -patch_size:]
        padding_mask = torch.zeros((B, num_patches), dtype=torch.int, device=x.device)
        padding_mask[:, -1] = 1  # Last position is always padded
        for i in range(B):
            for j in range(num_patches - 1, -1, -1):
                if torch.all(x[i, :, j * patch_size:(j + 1) * patch_size] == padding_reference[i, :, :]):
                    padding_mask[i, j] = 1
                else:
                    break
        return padding_mask




class MAE1DViT(nn.Module):
    """ Masked Autoencoder with Transformer backbone for 1D Time-Series """
    def __init__(self,
                 in_chans=len([0,1,2,3,4,5,6]),
                 input_size=2**18,  # 7.3 hours of monitoring @ 10Hz sampling rate: 216000
                 patch_size=1024,
                 embed_dim=512,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 loss_version='v1.0',
                 loss_channels=[2],
                 ch_weights=None,
                 # Features prediction parameters
                 feature_prediction_dim = 256,
                 feature_hidden_dims = [512, 256],
                 feature_dropout = 0.1,
                 age_loss_weight = 1.0,
                 sex_loss_weight = 1.0,
                 ahi_loss_weight = 1.0,
                 sqi_loss_weight = 1.0,
                 recon_loss_weight = 1.0
                 ):
        super().__init__()

        # Define number of patches based on sequence length and patch size
        self.patch_size = patch_size
        self.seq_length = input_size
        num_patches = input_size // patch_size
        self.in_chans = in_chans     #NUMBER OF INPUT CHANNELS
        self.loss_channels = loss_channels    #!! relative indices to the input channels indices!!
        if ch_weights is None:
            # Initialize ch_weights to uniformed weights only on the loss channels
            self.ch_weights = torch.zeros(in_chans)
            fill_value = 1 / len(loss_channels)
            self.ch_weights[self.loss_channels] = fill_value
        else:
            assert len(ch_weights) == len(loss_channels), "channel weights must fit number of loss channels"
            self.ch_weights = ch_weights

        # Define the loss version and weights
        self.loss_version = loss_version
        self.age_loss_weight = age_loss_weight
        self.sex_loss_weight = sex_loss_weight
        self.ahi_loss_weight = ahi_loss_weight
        self.sqi_loss_weight = sqi_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.dtw_loss = SoftDTWLossPyTorch(gamma=0.1, normalize=True)

        # MAE encoder specifics
        self.patch_embed = PatchEmbedding1D(self.in_chans,  self.patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # Positional embedding

        # Transformer encoder layers
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans,
                                      bias=True)  # Output layer for time-series patches

        # Features prediction heads
        self.age_predictor = self._build_predictor(embed_dim, feature_hidden_dims, feature_dropout)
        self.sex_predictor = self._build_predictor(embed_dim, feature_hidden_dims, feature_dropout)
        self.ahi_predictor = self._build_predictor(embed_dim, feature_hidden_dims, feature_dropout)
        self.sqi_predictor = self._build_predictor(embed_dim, feature_hidden_dims, feature_dropout)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def _build_predictor(self, input_dim, hidden_dims, dropout):
        """Build MLP for prediction (of age, sex etc..) from CLS token representation"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final output layer (single value for age)
        layers.append(nn.Linear(prev_dim, 1))

        return nn.Sequential(*layers)

    def initialize_weights(self):
        # Initialize positional embeddings (sin-cos 1D encoding)
        pos_embed = get_1d_sincos_pos_embed(
                        emb_dim= self.pos_embed.shape[-1], 
                        grid_size= (self.seq_length // self.patch_size),
                        cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(
                        emb_dim = self.decoder_pos_embed.shape[-1],
                        grid_size= (self.seq_length // self.patch_size),
                        cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize cls_token and mask_token
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize predictor weights
        for predictor in [self.age_predictor, self.sex_predictor, self.ahi_predictor, self.sqi_predictor]:
            for m in predictor.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)

    def patchify(self, x):
        """
        Converts time-series input into patches.
        x: (N, in_chans, seq_length)
        Returns: (N, num_patches, patch_size * in_chans)
        """
        N, C, L = x.view(x.size(0), x.size(1), -1).shape
        assert L == self.seq_length, "Input sequence length doesn't match model configuration."
        x = x.view(N, C, L // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(N, L // self.patch_size, -1)  # (N, num_patches, patch_size * in_chans)
        return x

    def get_padding_mask(self, x):
        """
        Identify padded regions in the input sequence, in a same way as previous function but in an unpacthified sequence.
        Args:
            x: Input tensor of shape [B, C, seq_len]
        Returns:
            padding_mask: Binary tensor [B, num_patches], where 1 marks padded patches, for a patchified sample
        """
        B, C, seq_len = x.shape
        patch_size = self.patch_size
        num_patches = seq_len // patch_size
        padding_reference = x[:, :, -patch_size:]
        padding_mask = torch.zeros((B, num_patches), dtype=torch.int, device=x.device)
        padding_mask[:, -1] = 1  # Last position is always padded
        for i in range(B):
            for j in range(num_patches - 1, -1, -1):
                if torch.all(x[i, :, j * patch_size:(j + 1) * patch_size] == padding_reference[i, :, :]):
                    padding_mask[i, j] = 1
                else:
                    break
        return padding_mask

    def random_masking(self, x, mask_ratio, padding_mask):
        """
        Perform per-sample random masking by per-sample shuffling, ensuring padded regions are always masked.
        Per-sample shuffling is done by argsort random noise.
        Args:
            x: Input tensor of shape [N, L, D] where:
               - N is the batch size,
               - L is the num_patches,
               - D is the embedding dimension.
            mask_ratio: Proportion of the sequence to mask.
            padding_mask: Binary tensor [N, L], where 1 marks padded patches.

        Returns:
            x_masked: Masked sequence with the specified ratio of positions masked.
            mask: Binary mask indicating masked positions (1 for masked, 0 for kept).
            ids_restore: Indices to restore the original sequence order.
                        padding_mask: Binary tensor [N, L], where 1 marks padded patches.

        """
        B, N, D = x.shape  # batch, number_patches, dim

        # Identify unpadded positions
        valid_mask = 1 - padding_mask  # 1 for valid patches, 0 for padded
        num_valid_patches = valid_mask.sum(dim=1)  # Number of unpadded patches per sample
        len_keep = (num_valid_patches * (1 - mask_ratio)).long()  # (B, ) Compute how many valid patches to keep
        len_keep = len_keep.min()  # Ensure all samples in batch have the same number of seen patches

        noise = torch.rand(B, N, device=x.device) * valid_mask  # Apply noise only to unpadded regions
        noise = noise + padding_mask  # Ensure padded regions remain masked
        # sort non-zero noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first `len_keep` unpadded patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # including the padded regions

        # Generate the binary mask: 0 is keep, 1 is mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def sequential_masking(self, x, mask_ratio, padding_mask, mask_type="beginning"):
        """
        Perform sequential masking (beginning or end) on the input, ensuring padded regions are always masked.

        Args:
            x: Input tensor of shape [N, L, D] where:
               - N is the batch size,
               - L is the num_patches,
               - D is the embedding dimension.
            mask_ratio: Proportion of the sequence to mask.
            padding_mask: Binary tensor [N, L], where 1 marks padded patches.
            mask_type: "beginning" or "end"

        Returns:
            x_masked: Masked sequence with the specified ratio of positions masked.
            mask: Binary mask indicating masked positions (0 for kept, 1 for masked).
            ids_restore: Indices to restore the original sequence order.
        """
        B, N, D = x.shape  # batch, number_patches, dim

        # Identify unpadded positions
        valid_mask = 1 - padding_mask  # 1 for valid patches, 0 for padded
        num_valid_patches = valid_mask.sum(dim=1)  # Number of unpadded patches per sample
        len_keep = (num_valid_patches * (1 - mask_ratio)).long()  # (B, ) Compute how many valid patches to keep
        len_keep = len_keep.min()  # Ensure all samples in batch have the same number of seen patches

        # Create sequential indices for each sample
        ids_shuffle = torch.zeros([B, N], device=x.device, dtype=torch.long)

        for b in range(B):
            valid_patches = num_valid_patches[b].item()

            if mask_type == "end":
                # Keep the first len_keep patches, then the remaining valid patches, then padding
                kept_indices = torch.arange(len_keep, device=x.device)
                masked_indices = torch.arange(len_keep, valid_patches, device=x.device)
            else:  # mask_type == "beginning"
                # Keep the last len_keep patches, then the remaining valid patches, then padding
                start_keep = valid_patches - len_keep
                kept_indices = torch.arange(start_keep, valid_patches, device=x.device)
                masked_indices = torch.arange(0, start_keep, device=x.device)

            # Create the shuffle order: kept patches first, then masked patches, then padding
            shuffle_order = torch.zeros(N, device=x.device, dtype=torch.long)
            shuffle_order[:len_keep] = kept_indices
            shuffle_order[len_keep:len_keep + len(masked_indices)] = masked_indices

            # Add padding indices at the end
            if valid_patches < N:
                padding_indices = torch.arange(valid_patches, N, device=x.device)
                shuffle_order[len_keep + len(masked_indices):] = padding_indices

            ids_shuffle[b] = shuffle_order

        # Compute ids_restore
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first `len_keep` patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def unpatchify(self, x):
        """
        Reconstruct time-series from patches.
        x: (Batch, num_patches, patch_size * in_chans)
        Returns: (Batch, in_chans, seq_length)
        """
        B, N, D = x.shape
        C = D // self.patch_size
        x = x.view(B, N, C, self.patch_size).permute(0, 2, 1, 3).reshape(B, C,
                                                                         N * self.patch_size)  # (N, in_chans, seq_length)
        return x
    
    def closest_power_of_2(self, n):
        if n <= 0:
            raise ValueError("Input must be a positive number")
        return 2 ** math.ceil(math.log2(n))

    def compute_reconstruction_loss(self, x, target, mask, padding_mask):
        """
        'v1.0': vanilla mse
        'v2.0': vanilla mse + variance
        'v3.1': weighted channels mse                   ! this the one to be used for MSE on a loss_channels only!
        'v3.2': weighted channels DTW
        'v3.3': weighted channels mse + DTW

        """
        # take the masked patches only from non-padded areas
        mask = mask * (1 - padding_mask)

        if self.loss_version == 'v1.0':
            ######### MSE on all inputs channels ######### (Vanilla loss)
            loss = (x - target) ** 2
            loss = loss.mean(dim=-1)  # [B, n_patches], mean loss per patch
            loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches

        elif self.loss_version == 'v2.0':
            ######### MSE + variance on all inputs channels#########
            mse_loss = (x - target) ** 2
            mse_loss = mse_loss.mean(dim=-1)  # [B, n_patches], mean loss per patch
            var_loss = (x.var(dim=-1) - target.var(dim=-1)) ** 2   # compare variance of each patch
            loss = (mse_loss + var_loss)/2
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        elif self.loss_version.startswith("v3."):
            ######### per channel loss #########

            # dimensions:
            B, num_of_patches, patch_size = x.shape  # (B, n_patches, patch_size*channels)
            patch_seq_len = int(patch_size / self.in_chans)

            ### compute MSE loss per channel ###
            se_patchified_loss = (x - target) ** 2  # shape: (B, num_patches, patch_size*channels)
            se_unpatchified_loss = self.unpatchify(se_patchified_loss)  # shape: (B, in_ch, seq_len)
            mask_unpatchified = torch.repeat_interleave(mask, patch_seq_len, axis=1)
            mask_unpatchified = torch.tensor(mask_unpatchified, device=se_unpatchified_loss.device).bool() #shape: (B, seq_len)
            # keep only masked MSE
            mse_loss = torch.stack([se_unpatchified_loss[i, :, mask_unpatchified[i]].mean(dim=-1) for i in range(B)])  # shape: (B, in_chan,)

            if self.loss_version.split('.')[1] != '1':  # v3.2/3
                ### compute DTW loss per channel using self.dtw_loss ###
                dtw_patchified_loss = torch.zeros((B, self.in_chans), device=x.device)
                mask_expanded = mask[:, :, None, None].expand(-1, -1, self.in_chans, patch_seq_len)

                x_patchified = x.view(B, num_of_patches, self.in_chans, patch_seq_len)
                target_patchified = target.view(B, num_of_patches, self.in_chans, patch_seq_len)

                for b in range(B):  # loop over batch
                    for ch in range(self.in_chans):  # loop over channels
                        # extract masked patches for current sample and channel
                        cur_mask = mask_expanded[b, :, ch, 0].bool()  # shape: (num_of_patches,)
                        cur_x = x_patchified[b, cur_mask, ch, :]  # shape: (n_masked_patches, patch_seq_len)
                        cur_target = target_patchified[b, cur_mask, ch, :]  # shape: (n_masked_patches, patch_seq_len)

                        if cur_x.shape[0] == 0:
                            dtw_patchified_loss[b, ch] = 0.0
                        else:
                            dtw_patchified_loss[b, ch] = self.dtw_loss(cur_x.squeeze().unsqueeze(0), cur_target.squeeze().unsqueeze(0))

                dtw_loss = dtw_patchified_loss  # shape: (B, in_chans)

            ####### Following section to be doubled checked !!
            # preferable to compute FFT/Var loss per patch then aggregate

            # ### compute VARIANCE LOSS per channel ###
            # unpatchified_masked_x = self.unpatchify(masked_x)  # shape: (B, in_ch, masked_seq_len)
            # unpatchified_target_x = self.unpatchify(masked_target)  # shape: (B, in_ch, masked_seq_len)
            # var_loss = (unpatchified_masked_x.var(dim=-1) - unpatchified_target_x.var(dim=-1)) ** 2  # shape: (B, in_chans)
            #
            # ### compute the FFT loss per channel ###
            # n = self.closest_power_of_2(unpatchified_masked_x.shape[2])  # find the closet power of 2 of the masked_seq_len
            # ch_x_fft_1d = torch.fft.fft(unpatchified_masked_x.to(torch.float32), n=n, dim=-1)
            # ch_target_fft_1d = torch.fft.fft(unpatchified_masked_x.to(torch.float32), n=n, dim=-1)
            # ch_fft_loss = (torch.abs(ch_x_fft_1d) - torch.abs(ch_target_fft_1d)) ** 2
            # weight = 1/n # normalize the loss based on patch size
            # fft_loss = weight*ch_fft_loss.mean(dim=(0, 1, 3))

            #######

            # split the loss_version and check the number after the second point, and apply MSE or MSE + variance loss respectively
            if self.loss_version.split('.')[1] == '1':   # v3.1
                loss = mse_loss
            elif self.loss_version.split('.')[1] == '2':   # v3.2
                loss = dtw_loss
            elif self.loss_version.split('.')[1] == '3':   # v3.3
                loss = (mse_loss + dtw_loss)/2


            ########## Appply channels weights to loss #########
            # Initialize weights:
            weights = torch.tensor(self.ch_weights, device=loss.device)
            # resize the weights to be (loss.shape[0], self.ch_weights) to match the loss shape and so that it would apply on the 2nd dim
            weights = weights.unsqueeze(0).repeat(loss.shape[0], 1)
            # Remove NaN and Inf values from loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                loss = torch.where(torch.isnan(loss), torch.tensor(0.0, device=loss.device), loss)
            # Compute weighted loss
            loss = (loss * weights).sum(axis=1)              # !!!!! check cases where weights is not  == 1
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # If any loss is NaN, print a warning and exit
                print(f"Warning: NaN detected in loss computation for version {self.loss_version}.")
                print("loss: ", loss)
                print(loss.shape)
                # print all data
                # print("mse_loss: ", mse_loss)
                # print(mse_loss.shape)
                # print("weights:", weights)
                # print(weights.shape)
                # print("loss*weights: ", (loss * weights))
                # print((loss*weights).shape)

        else:
            print("no available loss version")
            sys.exit()

        return loss.mean()  # average on batch

    def forward_encoder(self, x, mask_ratio, various_masking_strategies=False, mask=None, ids_restore=None, ids_keep=None):

        # determine padding_mask from unpatchified input
        padding_mask = self.get_padding_mask(x)

        # Patchify input and project into latent space
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # Add positional embedding
        try:
            x = x + self.pos_embed[:, 1:, :]
        except RuntimeError as e:
            print(e)
            print(f"shape of x: {x.shape}")
            print(f"shape of pos_embeded: {self.pos_embed.shape}")
            sys.exit()

        if mask is not None:
            # Apply input mask
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        elif various_masking_strategies:
            # Randomly choose masking strategy with 1/3 probability each
            rand_val = torch.rand(1).item()
            if rand_val < 1 / 3:
                # Random masking (original implementation)
                x, mask, ids_restore = self.random_masking(x, mask_ratio, padding_mask)
            elif rand_val < 2 / 3:
                # Mask beginning of time series
                x, mask, ids_restore = self.sequential_masking(x, mask_ratio, padding_mask, mask_type="beginning")
            else:
                # Mask end of time series
                x, mask, ids_restore = self.sequential_masking(x, mask_ratio, padding_mask, mask_type="end")
        else:
            # Default to random masking
            x, mask, ids_restore = self.random_masking(x, mask_ratio, padding_mask)

        # Add cls token
        cls_tokens = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # latent representations
        latent = x.clone()

        return x, mask, ids_restore, latent, padding_mask
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)  # embedding tokens
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append cls token
        # add positional embedding
        x = x + self.decoder_pos_embed
        # apply transformer decoder
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # Remove cls token
        x = x[:, 1:, :]                      # (B, num_patches, patch_size * in_chans)

        return x

    def forward_predictions(self, latent):
        """
        Forward pass for predictions using CLS token
        latent: (B, num_patches+1, embed_dim) - includes CLS token at position 0
        """
        cls_token = latent[:, 0, :]  # Extract CLS token (B, embed_dim)
        age_pred = self.age_predictor(cls_token)  # (B, 1)
        sex_pred = self.sex_predictor(cls_token)  # (B, 1)
        ahi_pred = self.ahi_predictor(cls_token)  # (B, 1)
        sqi_pred = self.sqi_predictor(cls_token)  # (B, 1)
        return age_pred.squeeze(-1), sex_pred.squeeze(-1) , ahi_pred.squeeze(-1), sqi_pred.squeeze(-1)  # (B,) - remove last dimension

    def compute_regression_loss(self, pred, target, reduction='mean'):
        """
        Functional version of MSE loss that ignores missing values.
        Args:
            pred: Predicted values (tensor)
            target: Target values (tensor)
            reduction (str): 'none' | 'mean' | 'sum'
        Returns:
            MSE loss with missing values ignored, after reduction
        """
        if target is None or not (all(isinstance(x, float) for x in target)):
            return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)


        # Create mask for valid (non-NaN) values
        target = torch.tensor([float('nan') if (x is None) else x for x in target], device=pred.device)
        valid_mask = ~(torch.isnan(pred) | torch.isnan(target))

        # If no valid values, return NaN
        if not valid_mask.any():
            return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)

        # Extract valid values and compute MSE
        valid_pred = pred[valid_mask]
        valid_target = torch.tensor(target[valid_mask], device=pred.device)
        squared_errors = (valid_pred - valid_target) ** 2

        if reduction == 'none':
            result = torch.full_like(pred, float('nan'))
            result[valid_mask] = squared_errors
            return result
        elif reduction == 'mean':
            return squared_errors.mean()
        elif reduction == 'sum':
            return squared_errors.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def compute_classification_loss(self, pred, target, reduction='mean'):
        """
        Functional version of Binary Cross-Entropy loss that ignores missing values.
        Args:
            pred: Predicted values (tensor) - logits or probabilities
            target: Target values (tensor) - binary labels (0 or 1)
            reduction (str): 'none' | 'mean' | 'sum'
        Returns:
            BCE loss with missing values ignored, after reduction
        """
        if target is None or not (all(isinstance(x, float) for x in target)):
            return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)

        # Create mask for valid (non-NaN) values
        target = torch.tensor([float('nan') if x is None else x for x in target], device=pred.device)
        valid_mask = ~(torch.isnan(pred) | torch.isnan(target))

        # If no valid values, return NaN
        if not valid_mask.any():
            return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)

        # Extract valid values and compute BCE
        valid_pred = pred[valid_mask]
        valid_target = torch.tensor(target[valid_mask], device=pred.device)

        # Compute binary cross-entropy loss
        # Using F.binary_cross_entropy_with_logits for numerical stability if pred contains logits
        # Or F.binary_cross_entropy if pred contains probabilities
        return nn.functional.binary_cross_entropy_with_logits(valid_pred, valid_target, reduction=reduction)


    def forward_loss(self, inputs, x, mask, padding_mask,
                     age_pred=None, age_target=None,
                     sex_pred=None, sex_target=None,
                     ahi_pred=None, ahi_target=None,
                     sqi_pred=None, sqi_target=None
                     ):
        """
        inputs: input tensor (B, C, seq_length)
        x: reconstructed tensor (B, num_patches, patch_size * in_chans)
        mask: binary mask indicating masked positions (1 for masked, 0 for kept)
        feature_pred: predicted values (B,)
        feature_target: targets (B,)
        """
        # Reconstruction loss
        target = self.patchify(inputs)  # (B, num_patches, patch_size * in_chans)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        recon_loss = self.compute_reconstruction_loss(x, target, mask, padding_mask)

        # Age prediction loss
        if age_pred is not None and age_target is not None:
            age_loss = self.compute_regression_loss(age_pred, age_target)
        else:
            age_loss = torch.tensor(0.0, device=x.device)
        # Sex prediction loss
        if sex_pred is not None and sex_target is not None:
            sex_loss = self.compute_classification_loss(sex_pred, sex_target)
        else:
            sex_loss = torch.tensor(0.0, device=x.device)
        # AHI prediction loss
        if ahi_pred is not None and ahi_target is not None:
            ahi_loss = self.compute_regression_loss(ahi_pred, ahi_target)
        else:
            ahi_loss = torch.tensor(0.0, device=x.device)
        # SQI prediction loss
        if sqi_pred is not None and sqi_target is not None:
            sqi_loss = self.compute_regression_loss(sqi_pred, sqi_target)
        else:
            sqi_loss = torch.tensor(0.0, device=x.device)

        # Combined loss
        total_loss = (self.recon_loss_weight * recon_loss +
                      self.age_loss_weight * age_loss +
                      self.sex_loss_weight * sex_loss +
                      self.ahi_loss_weight * ahi_loss +
                      self.sqi_loss_weight * sqi_loss
                      )

        return total_loss, recon_loss, age_loss, sex_loss, ahi_loss, sqi_loss, target

    def forward(self, samples, mask_ratio=0.65, various_masking_strategies=False,
                mask=None, ids_restore=None, ids_keep=None,
                age_target=None, sex_target=None, ahi_target=None, sqi_target=None):
        """
        :param x: input tensor (B, C, seq_length)
        :return:
             loss - according to the version defined
             reconstructed patches - (B, num_patches, patch_size * in_chans)
             mask - vector of length num_patches with 1 for masked positions and 0 for kept positions
             latent - vector of embeddings representations as encoded by the encoder
             target - patchified input (B, num_patches, patch_size * in_chans)
        """
        # Encoder processing:
        emb_enc, mask, ids_restore, latent, padding_mask = self.forward_encoder(samples, mask_ratio=mask_ratio,
                                                                  various_masking_strategies=various_masking_strategies,
                                                                  mask=mask, ids_restore=ids_restore, ids_keep=ids_keep)
        if torch.isnan(emb_enc).any():
            print(f"nan detected in emb_enc after forward_encoder")

        # Decoder processing:
        pred = self.forward_decoder(emb_enc, ids_restore)

        if torch.isnan(pred).any():
            print(f"nan detected in pred after forward_decoder")

        # Predictions from CLS token
        if sex_target is not None:
            age_pred, sex_pred, ahi_pred, sqi_pred = self.forward_predictions(latent)
            if torch.isnan(age_pred).any():
                print("nan detected in age_pred after forward_predictions")
            if torch.isnan(sex_pred).any():
                print("nan detected in sex_pred after forward_predictions")
            if torch.isnan(ahi_pred).any():
                print("nan detected in ahi_pred after forward_predictions")
            if torch.isnan(sqi_pred).any():
                print("nan detected in sqi_pred after forward_predictions")
        else:
            age_pred, sex_pred, ahi_pred, sqi_pred = None, None, None, None

        # compute multi-loss
        # print(f"samples: {samples.shape}, pred: {pred.shape}, mask: {mask.shape}, "
        #       f"padding_mask: {padding_mask.shape}")
        total_loss, recon_loss, age_loss, sex_loss, ahi_loss, sqi_loss, target = self.forward_loss(
            samples, pred, mask, padding_mask,
            age_pred=age_pred, age_target=age_target,
            sex_pred=sex_pred, sex_target=sex_target,
            ahi_pred=ahi_pred, ahi_target=ahi_target,
            sqi_pred=sqi_pred, sqi_target=sqi_target)
        if torch.isnan(total_loss).any():
            print(f"nan detected in total_loss after forward_loss")

        return total_loss, pred, mask, padding_mask, latent, target, recon_loss, age_loss, sex_loss, ahi_loss, sqi_loss


class SleepSpectogramMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, input_size = (64, 4096), patch_size=(4,64), 
                 stride=10, in_chans=7,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 audio_exp=False, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 use_custom_patch=False, split_pos=False, pos_trainable=False, use_nce=False, beta=4.0, decoder_mode=0,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, no_shift=False, loss_version='v1.0',loss_channels = [0,1,2,3,4,5,6],  ch_weights=None
                 ):
        super().__init__()

        self.audio_exp=audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.freq_bins = input_size[0]
        self.time_frames = input_size[1]
        self.in_chans = in_chans
        img_size = self.freq_bins*self.time_frames
        self.use_custom_patch =use_custom_patch
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # if use_custom_patch:
        #     print(f'Use custom patch_emb with patch size: {patch_size}, stride: {stride}')
        #     self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        # else:
        #     self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        # self.use_custom_patch = use_custom_patch

        # use inhouse patch embeding
        self.patch_embed = PatchEmbedding2D(in_chans,self.freq_bins, self.time_frames, patch_size, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding


        self.no_shift=no_shift


        self.decoder_mode = decoder_mode
        if self.use_custom_patch: # overlapped patches as in AST. Similar performance yet compute heavy
            window_size= (6,6)
            feat_size = (102,12)
        else:
            window_size= (4,4)
            feat_size = (64,8)       #NEED TO CHECK         
        if self.decoder_mode == 1:
            decoder_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0,0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0,0)
                    else:
                        shift_size = (2,0)
                    #shift_size = tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size])
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        feat_size=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        drop_attn=0.0,
                        drop_path=0.0,
                        extra_norm=False,
                        sequential_attn=False,
                        norm_layer=norm_layer, #nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)        
        else:
            # Transfomer
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size[0] * patch_size[1], bias=True) # decoder to patch, OLD - PATCH_SIZE**2

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax=nn.LogSoftmax(dim=-1)

        self.mask_t_prob=mask_t_prob
        self.mask_f_prob=mask_f_prob
        self.mask_2d=mask_2d

        self.epoch = epoch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # EMBEDDING DIM FOR SPECTOGRAM (NON-SQUARE IMAGE)
        grid_width = self.freq_bins // self.patch_size[0]  # Calculate width based on patch size
        grid_height = self.time_frames // self.patch_size[1]  # Calculate height based on patch size

        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)    
        else:
            #ADAPTED
            pos_embed = get_2d_sincos_pos_embed_spectogram(
                embed_dim=self.pos_embed.shape[-1],
                grid_width=grid_width,
                grid_height=grid_height,
                cls_token=True
            )
            #old:
            # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True) #POS_EMBEDED (0,13690)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.audio_exp:   
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        else:
            #ADAPTED
            decoder_pos_embed = get_2d_sincos_pos_embed_spectogram(
                embed_dim=self.decoder_pos_embed.shape[-1],
                grid_width=grid_width,
                grid_height=grid_height,
                cls_token=True
            )
            #OLD
            #decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        print("decoder_pos_embed shape:", decoder_pos_embed.shape)  # Should be [1 + grid_width * grid_height, embed_dim]
        print("self.decoder_pos_embed shape:", self.decoder_pos_embed.shape)  # Should match [1, 1 + grid_width * grid_height, embed_dim]

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data #shape: (emb_dim, in_chans, patch_size[0], patch_size[1])
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W) #batch_size, channels, freuency, time*mask
        x: (B, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        B, CH, F, T = imgs.shape
        p1 = self.patch_embed.patch_size[0] 
        p2 = self.patch_embed.patch_size[1]
        num_patches = self.patch_embed.num_patches
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        if self.audio_exp:
            if self.use_custom_patch: # overlapped patch
                h,w = self.patch_embed.patch_hw
                # todo: fixed h/w patch size and stride size. Make hw custom in the future
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h*w, p**2 * 1))
                #x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                #x = torch.einsum('nchpwq->nhwpqc', x)
                #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                #h,w = self.patch_embed.patch_hw
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum('nchpwq->nhwpqc', x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else: 
            # h = w = imgs.shape[2] // p
            # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            # x = torch.einsum('nchpwq->nhwpqc', x)
            # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
            #[4, 7, 12, 4, 64, 64] = 5505024, 5849088
            h = F // p1
            w = T // p2
            x = imgs.reshape(shape=(imgs.shape[0], CH, h, p1, w, p2))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p1 * p2 * CH))

        return x

    def unpatchify(self, x):
        """
        x: (batch_size, num_of_patches, patch_size_height*patch_size_width * in_chans)B,1024, 1792
        specs: (N, 1, H, W)
        """
        patch_size = self.patch_embed.patch_size
        p1 = patch_size[0]
        p2 = patch_size[1]
        h = self.freq_bins//p1
        w = self.time_frames//p2
        x = x.reshape(shape=(x.shape[0], h, w, p1, p2, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x) #B, ch, h, p1, w, p2
        specs = x.reshape(shape=(x.shape[0],self.in_chans, h * p1, w * p2))
        return specs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, num_of_patches, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        T = self.time_frames
        F = self.freq_bins
        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask_2d=False, mask=None, ids_restore=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        try:
            x = x + self.pos_embed[:, 1:, :]
        except RuntimeError as e:
            print(e)
            print(f"shape of x: {x.shape}")
            print(f"shape of pos_embeded: {self.pos_embed.shape}")
        # masking: length -> length * mask_ratio
        # random_masking_2d????
        if mask is not None:
            # Apply input mask
            x = x * mask.unsqueeze(-1)
        else:
            # apply random masking
            if mask_2d:
                x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #emb = self.encoder_emb(x)
        latent = x.clone()
        return x, mask, ids_restore, latent

    def forward_encoder_no_mask(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        #x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        contextual_embs=[]
        for n, blk in enumerate(self.blocks):
            x = blk(x)
            if n > self.contextual_depth:
                contextual_embs.append(self.norm(x))
        #x = self.norm(x)
        contextual_emb = torch.stack(contextual_embs,dim=0).mean(dim=0)

        return contextual_emb

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        if self.decoder_mode != 0:
            B,L,D=x.shape
            x = x[:,1:,:]
            if self.use_custom_patch:
                x = x.reshape(B,101,12,D)
                x = torch.cat([x,x[:,-1,:].unsqueeze(1)],dim=1) # hack
                x = x.reshape(B,1224,D)
        if self.decoder_mode > 3: # mvit
            x = self.decoder_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B,102,12,256)
                pred = pred[:,:101,:,:]
                pred = pred.reshape(B,1212,256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]
        return pred, None, None #emb, emb_pixel

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, target      

    def forward(self, imgs, mask_ratio=0.8, mask=None, ids_restore=None):
        emb_enc, mask, ids_restore, latent = self.forward_encoder(imgs, mask_ratio, mask_2d=self.mask_2d,
                                                                  mask=mask, ids_restore=ids_restore)
        pred, _, _ = self.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*ch]
        loss_recon, target = self.forward_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)
        loss_contrastive = torch.FloatTensor([0.0]).cuda()

        return loss_recon, pred, mask, latent, target


# check defult parametres
def sleep_1d_original(**kwargs):
    model = MAE1DViT(
        depth=24, num_heads=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# spectogram: shape: (BATCH, CHANNELS, (h,w)))
def sleep_spec_demo_1(**kwargs):
    model = SleepSpectogramMAE(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b # decoder: 512 dim, 8 blocks
sleep_spec_demo = sleep_spec_demo_1
sleep_1d = sleep_1d_original
