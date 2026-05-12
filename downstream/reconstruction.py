import torch

from model.utils import *
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from tslearn.metrics import SoftDTWLossPyTorch


def display_reconstruction_MAEspec(inputs, outputs, masks, patch_size, loss_channels, relative_loss_channels, unpatchify): #input: original samples, outputs: reconstructed image
    batch_size, freq, time = inputs[0].shape
    input = inputs[0].cpu().numpy() # SHAPE (IN_CHANS, SEQ_LEN). 2d: (ch, freq,time),
    output = unpatchify(outputs)[0].cpu().detach().numpy() #(IN_CHANS, SEQ_LEN), 2d: (ch, freq,time)
    #START DEBUGGING FROM HERE
    num_patches_freq = freq // patch_size[0]
    num_patches_time = time // patch_size[1]
    mask = masks[0].reshape(num_patches_freq, num_patches_time)
    mask = mask.int().cpu().numpy()
    mask = np.repeat(np.repeat(mask, patch_size[0], axis=0), patch_size[1], axis=1)

    time_limit = 256
    plt.figure(figsize=(10, 10))
    for i, ch in enumerate(loss_channels):
        ax1 = plt.subplot(len(loss_channels), 1, i + 1)
        # extract channel
        displayed_input = input[relative_loss_channels[i], :, time_limit:time_limit*2]
        displayed_output = output[relative_loss_channels[i], :, :]
        # don't show output of unmasked parts
        displayed_output[~mask.astype(bool)] = np.nan
        displayed_output = displayed_output[:, time_limit:time_limit*2]
        ax1.imshow(displayed_input, cmap='Blues', interpolation='nearest')
        ax1.set_title(f"Input Image: {SIGNALS[ch]}", fontsize=12)
        ax1.axis('off')

        ax1.imshow(displayed_output, cmap='Reds', interpolation='nearest', alpha=0.5)
        ax1.set_title(f"Reconstructed Image: {SIGNALS[ch]}", fontsize=12)
        ax1.axis('off')

        if i == 0:
            ax1.set_title("Masked Input and Reconstructed Output", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PERSONAL_DIR, 'spectograms', f"reconstruction_{wandb.run.id}_full_night.png"))

    # plt.figure(figsize=(12, len(loss_channels) * 4))  # Adjust figure size

    # for i, ch in enumerate(loss_channels):
    #     # Subplot for input
    #     ax1 = plt.subplot(len(loss_channels), 2, i * 2 + 1)
    #     ax1.imshow(input[relative_loss_channels[i], :, :time_limit], cmap='Blues', interpolation='nearest')
    #     ax1.set_title(f"Input: {SIGNALS[ch]}", fontsize=12)
    #     ax1.axis('off')

    #     # Subplot for reconstructed output
    #     ax2 = plt.subplot(len(loss_channels), 2, i * 2 + 2)
    #     displayed_output = output[relative_loss_channels[i], :, :time_limit]
    #     displayed_output[~mask[:, :time_limit].astype(bool)] = np.nan  # Mask the unreconstructed parts
    #     ax2.imshow(displayed_output, cmap='Reds', interpolation='nearest')
    #     ax2.set_title(f"Reconstructed: {SIGNALS[ch]}", fontsize=12)
    #     ax2.axis('off')

    # plt.tight_layout()
    # plt.savefig(os.path.join(PERSONAL_DIR, 'spectograms', f"reconstruction_{wandb.run.id}_256timepoints.png"))
    # plt.show()

    # mins_to_display = 5
    # start_x = np.where(np.diff(mask, axis=2) == 1)[0][0] - (mins_to_display * 60 * RESAMPLE_RATE) / 2
    # end_x = start_x + (RESAMPLE_RATE * 60 * mins_to_display)
    # for i in range(len(loss_channels)):
    #     ax1 = plt.subplot(len(loss_channels), 1, i + 1)
    #     ax1.set_xlim([start_x, end_x])
    # ax1.set_xlabel(f"{mins_to_display} minute(s)", fontsize=12)
    # plt.savefig(os.path.join(PERSONAL_DIR, 'spectograms', f"reconstruction_{wandb.run.id}_{mins_to_display}min.png"))
    # plt.close()

def display_reconstruction_MAE1d(inputs, outputs, masks, patch_size, relative_loss_channels, unpatchify, task):
    input = inputs[0].cpu().numpy() # SHAPE (IN_CHANS, SEQ_LEN). 2d: (ch, freq,time)
    output = unpatchify(outputs)[0].cpu().detach().numpy() #(IN_CHANS, SEQ_LEN), 2d: (ch, freq,time)
    mask = np.repeat(masks[0].int().cpu().numpy(), patch_size)
    mask_ratio = mask.astype(float).mean()
    sampling_rate = int(wandb.run.config['data_path'].split("SR_")[1].split("Hz")[0])

    # handle case of input and output with not the same number of channels, in such case repeat the output
    if input.shape[0] != output.shape[0]:
        output = np.tile(output, (input.shape[0] // output.shape[0], 1))

    # full input length picture with residuals
    plt.figure(figsize=(10, 4 * len(relative_loss_channels)))
    # Iterate and display reconstruction and residuals for the LOSS channels
    for i, ch in enumerate(relative_loss_channels):
        # Extract data
        displayed_input = input[relative_loss_channels[i], :].flatten()
        displayed_output = output[relative_loss_channels[i], :].flatten()
        displayed_output[~mask.astype(bool)] = np.nan  # Mask the unreconstructed parts
        displayed_residuals = displayed_input - displayed_output
        displayed_residuals[~mask.astype(bool)] = np.nan
        time_axis = np.arange(len(displayed_input)) / sampling_rate
        patch_boundaries = np.arange(0, len(displayed_input), patch_size) / sampling_rate

        # Subplot 1: Original and Reconstructed
        ax1 = plt.subplot(len(relative_loss_channels) * 2, 1, 2 * i + 1)
        # --- Add grey background for reconstructed patches ---
        in_patch = False
        start_time = 0
        for t, m in zip(time_axis, mask):
            if m and not in_patch:  # start of reconstructed region
                start_time = t
                in_patch = True
            elif not m and in_patch:  # end of reconstructed region
                ax1.axvspan(start_time, t, color='silver', alpha=0.5, zorder=0)
                in_patch = False
        # if mask ends with a reconstructed patch
        if in_patch:
            ax1.axvspan(start_time, time_axis[-1], color='silver', alpha=0.5, zorder=0)
        # plot signals
        ax1.plot(time_axis, displayed_input, label="Original Signal", color="blue")
        ax1.plot(time_axis, displayed_output, label="Reconstructed Signal", color="red", alpha=0.5)
        ax1.set_ylabel(f"{SIGNALS[ch]}", fontsize=12)
        ax1.set_ylim([displayed_input.min(), displayed_input.max()])
        # Add vertical patch lines
        if len(patch_boundaries) < 60:
            for x in patch_boundaries:
                ax1.axvline(x=x, color="black", linestyle="dotted", linewidth=1.0)
        if i == 0:
            ax1.set_title(f"Original and Reconstructed Signals with Residuals\n(Masking Ratio: {mask_ratio:.0%})",
                          fontsize=14)
        ax1.legend(loc="lower left")

        # Subplot 2: Residuals
        ax2 = plt.subplot(len(relative_loss_channels) * 2, 1, 2 * i + 2)
        # --- Same grey background for residuals plot ---
        in_patch = False
        start_time = 0
        for t, m in zip(time_axis, mask):
            if m and not in_patch:
                start_time = t
                in_patch = True
            elif not m and in_patch:
                ax2.axvspan(start_time, t, color='silver', alpha=0.5, zorder=0)
                in_patch = False
        if in_patch:
            ax2.axvspan(start_time, time_axis[-1], color='silver', alpha=0.5, zorder=0)
        # plot residuals
        ax2.plot(time_axis, displayed_residuals, label="Residuals", color="black")
        ax2.set_ylim([displayed_input.min(), displayed_input.max()])
        ax2.set_ylabel("Residuals", fontsize=12)
        ax2.set_xlabel("Time (seconds)")
        # Add vertical patch lines
        if len(patch_boundaries) < 60:
            for x in patch_boundaries:
                ax2.axvline(x=x, color="black", linestyle="dotted", linewidth=1.0)
        ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(PERSONAL_DIR, 'figures', f"reconstruction_{wandb.run.id}_full_input_{task}_{mask_ratio}.png"))
    plt.close()

    #  check if the max x-axis in the previous plot was in hours
    max_x = time_axis.max()
    mins_to_display = 0.5
    if max_x >= 60:  # 60 sec
        # Zoom-in figure (one reconstructed patch)
        try:
            # Find first masked patch start and end
            mask_diff = np.diff(mask.astype(int))
            mask_start_idx = np.where(mask_diff == 1)[0][0]
            mask_end_idx = np.where(mask_diff == -1)[0][0]
            if mask_end_idx < mask_start_idx:
                # take the second mask
                mask_start_idx = np.where(mask_diff == 1)[0][1]
                mask_end_idx = np.where(mask_diff == -1)[0][2]

            # Compute time axis (in seconds)
            time_axis = np.arange(len(mask)) / sampling_rate

            # Compute patch boundaries in seconds
            patch_boundaries = np.arange(0, len(mask), patch_size) / sampling_rate

            start_idx = mask_start_idx - int((mins_to_display * 60 * sampling_rate) / 2)
            end_idx = start_idx + int(mins_to_display * 60 * sampling_rate)
            # Plot
            plt.figure(figsize=(10, 4 * len(relative_loss_channels)))
            for i, ch in enumerate(relative_loss_channels):
                ax = plt.subplot(len(relative_loss_channels), 1, i + 1)

                # Extract signal
                displayed_input = input[relative_loss_channels[i], :].flatten()
                displayed_output = output[relative_loss_channels[i], :].flatten()
                displayed_output[~mask.astype(bool)] = np.nan
                displayed_residuals = displayed_input - displayed_output
                displayed_residuals[~mask.astype(bool)] = np.nan

                # Plot signals
                ax.plot(time_axis[start_idx:end_idx], displayed_input[start_idx:end_idx], label="Original Signal",
                        color="blue")
                ax.plot(time_axis[start_idx:end_idx], displayed_output[start_idx:end_idx], label="Reconstructed Signal",
                        color="red", alpha=0.5)
                ax.set_ylabel(f"{SIGNALS[ch]}", fontsize=12)
                ax.set_xlim([time_axis[start_idx], time_axis[end_idx]])
                ax.set_ylim([displayed_input.min(), displayed_input.max()])

                # Add vertical dotted patch boundary lines
                for x in patch_boundaries:
                    if time_axis[start_idx] <= x <= time_axis[end_idx]:
                        ax.axvline(x=x, color="black", linestyle="dotted", linewidth=0.8)
                # --- Add grey background for reconstructed patches ---
                in_patch = False
                start_time = 0
                for t, m in zip(time_axis, mask):
                    if m and not in_patch:  # start of reconstructed region
                        start_time = t
                        in_patch = True
                    elif not m and in_patch:  # end of reconstructed region
                        ax.axvspan(start_time, t, color='silver', alpha=0.5, zorder=0)
                        in_patch = False
                # if mask ends with a reconstructed patch
                if in_patch:
                    ax.axvspan(start_time, time_axis[-1], color='silver', alpha=0.5, zorder=0)
                if i == 0:
                    ax.set_title(f"Zoomed View: ±{int(mins_to_display*60/2)} Seconds Around First Patch Boundary"
                                 f"\nMasking Ratio: {mask_ratio:.0%}")

                ax.legend(loc="lower left")

            plt.xlabel("Time (seconds)", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(PERSONAL_DIR, 'figures', f"reconstruction_{wandb.run.id}_zoom_{int(mins_to_display*60/2)}sec_{task}_{mask_ratio}.png"))
            plt.close()

        except Exception as e:
            print(f"No unmasked area found in non-padded area: {e}")


def pcc(x, y):
    """ Computes the similarity between two tensors using Pearson correlation
    along the last dimension N, where x and y are tensors of shape (B, C, N).
    """
    # Normalize along sequence/patch dimension
    x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
    y_norm = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True) + 1e-6)

    # Calculate cross-correlation along sequence dimension
    correlation = torch.sum(x_norm * y_norm, dim=-1) / x.shape[-1]
    return correlation  # Shape: (B, C)

def max_cross_correlation_fft(x, y):
    """
    Compute the maximal cross-correlation between two tensors over the last dimension using FFT
    """
    # Ensure inputs have the same shape
    assert x.shape == y.shape, "Tensors must have the same shape"

    # Convert to float32 (or float64) to avoid FFT dtype errors
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    # Normalize: Compute Z-scored version of x and y along the last dimension
    x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, unbiased=False, keepdim=True).clamp(min=1e-8)
    y = (y - y.mean(dim=-1, keepdim=True)) / y.std(dim=-1, unbiased=False, keepdim=True).clamp(min=1e-8)

    T = x.shape[-1]  # Time dimension

    # Compute FFT of x and y (complex-valued result)
    x_fft = torch.fft.rfft(x, dim=-1)
    y_fft = torch.fft.rfft(y, dim=-1)

    # Compute cross-correlation in frequency domain
    cross_corr_fft = x_fft * torch.conj(y_fft)  # Element-wise complex multiplication

    # Inverse FFT to get cross-correlation in time domain
    cross_corr = torch.fft.irfft(cross_corr_fft, n=T, dim=-1) / T

    # Find maximum correlation across all shifts
    max_corr = cross_corr.max(dim=-1).values  # Shape: (...,)

    return max_corr

def pcc_on_masked_patches(x, target, mask, relative_loss_channels, unpatchify_fn, patchify_fn):  # outputs, target
    """
    Compute the Pearson correlation between the reconstructed versus actual masked patches (concatenated)
    and the cross-correlation of each patch with its corresponding target patch.
    Args:
        x: patchified reconstructed X,
        target: patchified sample,
        mask: shape(B, num_of_patches)
        relative_loss_channels: The channels to evaluate the Pearson r on
        unpatchify_fn: The function to unpatchify the patches
        patchify_fn: The function to patchify the data
    Returns:
        (tensor) |Pearson r| per patch averaged over the batch and all channels
        (tensor) xcorr per patch averaged over the batch, channels and all masked patches
        (tensor) DTW per patch averaged over the batch, channels and all masked patches
    """
    # extract dims
    B, num_of_patches, patch_size = x.shape  # (B, n_patches, patch_size*channels)

    # unpatchify to get per channel data
    x_per_ch = unpatchify_fn(x)  # shape: (B, in_chans, seq_len)
    target_per_ch = unpatchify_fn(target)  # shape: (B, in_chans, seq_len)
    patch_seq_len = int(patch_size / x_per_ch.shape[1])

    # remove batch that don't have the same number of masked patches
    valid_batch_indices = [i for i in range(B) if mask[i].sum() == mask.sum(1).median()]
    mask = mask[valid_batch_indices, :]
    B = len(valid_batch_indices)

    corrs = []
    xcorrs = []
    dtws = []
    if x_per_ch.shape[1] == 1:
        relative_loss_channels = [0]
    for channel in relative_loss_channels:
        # extract only loss channels values
        X = x_per_ch[valid_batch_indices, channel, :].unsqueeze(1)
        Y = target_per_ch[valid_batch_indices, channel, :].unsqueeze(1)

        # patchify back to get the original shape per patch
        X = patchify_fn(X)
        Y = patchify_fn(Y)  # shape: (B, n_patches, patch_size)

        # compute the cross-correlation of each patch with its corresponding target patch
        xcorr_patchified = max_cross_correlation_fft(X, Y).abs()  # shape: (B, num_masked_patches)
        # keep only masked xcorr, and average over them
        xcorr = torch.stack([xcorr_patchified[i, mask.bool()[i, :]].mean() for i in range(B)])  # shape: (B,) # mean over patches
        xcorrs.append(xcorr.mean())  # mean over batch

        # compute the DTW of each patch with its corresponding target patch
        dtw_loss = SoftDTWLossPyTorch(gamma=0.1, normalize=True)
        masked_X = torch.stack([X[i, mask.bool()[i, :], :] for i in range(B)])
        masked_Y = torch.stack([Y[i, mask.bool()[i, :], :] for i in range(B)])
        # dtw = dtw_loss(masked_X.float(), masked_Y.float())  # shape: (B, )
        # dtws.append(dtw.mean())  # mean over batch

        # compute the Pearson correlation on the concatenated masked patches
        corrs_in_batch = []
        for i in range(B):
            # keep only masked patches and concatenate them
            masked_x = x[i, mask[i].bool(), :].view(-1)  # shape: (num_masked_patches*patch_size, )
            masked_target = target[i, mask[i].bool(), :].view(-1)

            # compute the Pearson correlation
            corr = pcc(masked_x, masked_target).abs()  # shape: (1, )
            corrs_in_batch.append(corr)  # Absolute Pearson correlation per channel
        corrs.append(torch.stack(corrs_in_batch).mean())  # average over batch

    return torch.stack(corrs).mean(), torch.stack(xcorrs).mean(), torch.tensor(0.0) # torch.stack(dtws).mean() # average over loss channels



def random_masking_reconstruction(model,
                                  dataloader,
                                  device,
                                  input_channels,
                                  patch_size,
                                  loss_channels):
    """Loop over batches and evaluate model for different predefined
    mask ratios. Log the mean and s.d. of the MSE and Pearson correlation
    for each mask ratio.
    Args:
        model: The model to evaluate
        dataloader: The dataloader to loop over
        device: The device to run the model on
        input_channels: The input channels of the model
        patch_size: The patch size of the model
        loss_channels: The channels to evaluate the loss
    """
    # Initializations
    model.eval()
    mask_ratios_for_evaluation = sorted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # Ensure sorted order
    relative_loss_channels = [input_channels.index(ch) for ch in loss_channels]
    binary_masks = {}
    ids_restore = {}
    ids_keep = {}
    mse_dict = {}
    corr_dict = {}
    xcorr_dict = {}
    corr10_dict = {}
    xcorr10_dict = {}
    mse10_dict = {}
    dtw_dict = {}
    dtw10_dict = {}

    # extract first sample patches from dataloader
    with torch.no_grad():
        x = next(iter(dataloader))[0].to(device, non_blocking=True)
        x = model.patch_embed(x)  # (B, num_patches, embed_dim)

    mask_dir = os.path.join(PARENT_DIR, f'masks_for_{x.shape[1]}_patches')

    if not os.path.exists(mask_dir):
        # Generate all nested masks at once
        mkdirifnotexists(mask_dir)
        results = nested_random_masking(x, mask_ratios_for_evaluation)

        # Save all masks
        for mask_ratio in mask_ratios_for_evaluation:
            # save only the first sample mask, ids_keep and ids_restore vectors (will then be replicated across batch)
            binary_masks[mask_ratio] = results[mask_ratio]['mask'][0]
            ids_restore[mask_ratio] = results[mask_ratio]['ids_restore'][0]
            ids_keep[mask_ratio] = results[mask_ratio]['ids_keep'][0]

            # Save binary mask
            save_binary_vector(binary_masks[mask_ratio],
                               os.path.join(mask_dir, f"mask_{mask_ratio}.bit"))

            # Save restore indices
            ids_restore_np = ids_restore[mask_ratio].cpu().detach().numpy()
            np.save(os.path.join(mask_dir, f"ids_restore_{mask_ratio}.npy"),
                    ids_restore_np)

            # Save keep indices
            ids_keep_np = ids_keep[mask_ratio].cpu().detach().numpy()
            np.save(os.path.join(mask_dir, f"ids_keep_{mask_ratio}.npy"),
                    ids_keep_np)
    else:
        # Load existing masks, ids_restore and ids_keep
        for mask_ratio in mask_ratios_for_evaluation:
            binary_masks[mask_ratio] = read_binary_vector(
                os.path.join(mask_dir, f"mask_{mask_ratio}.bit"),
                x.shape[1]
            )
            binary_masks[mask_ratio] = torch.tensor(binary_masks[mask_ratio]).to(device, non_blocking=True)

            ids_restore[mask_ratio] = np.load(
                os.path.join(mask_dir, f"ids_restore_{mask_ratio}.npy")
            )
            ids_restore[mask_ratio] = torch.tensor(ids_restore[mask_ratio]).to(device, non_blocking=True)

            ids_keep[mask_ratio] = np.load(
                os.path.join(mask_dir, f"ids_keep_{mask_ratio}.npy")
            )
            ids_keep[mask_ratio] = torch.tensor(ids_keep[mask_ratio]).to(device, non_blocking=True)

    # Initialize dictionaries for metrics
    for mask_ratio in mask_ratios_for_evaluation:
        mse_dict[mask_ratio] = []
        corr_dict[mask_ratio] = []
        xcorr_dict[mask_ratio] = []
        corr10_dict[mask_ratio] = []
        mse10_dict[mask_ratio] = []
        xcorr10_dict[mask_ratio] = []
        dtw_dict[mask_ratio] = []
        dtw10_dict[mask_ratio] = []

    # Evaluation on given dataset
    for data_iter_step, (samples, _ids) in enumerate(dataloader):
        with torch.no_grad():
            # compute validation loss (MSE) and pearson correlation for different masking ratios:
            for mask_ratio in mask_ratios_for_evaluation:
                # move samples to device
                samples = samples.to(device, non_blocking=True)

                # forward pass
                with torch.cuda.amp.autocast():
                    mse, outputs, masks, padding_mask,  _, target, _, _, _, _, _ = model(samples,
                                                                          mask=binary_masks[mask_ratio].repeat(samples.shape[0], 1).to(device),
                                                                          ids_restore=ids_restore[mask_ratio].repeat(samples.shape[0], 1).to(device),
                                                                          ids_keep=ids_keep[mask_ratio].repeat(samples.shape[0], 1).to(device))
                # save batch-level metrics
                masks = masks * (1 - padding_mask) # effective mask for metric calculation
                mse_dict[mask_ratio].append(mse.mean().item())
                corr, xcorr, dtw = pcc_on_masked_patches(outputs, target, masks,
                           relative_loss_channels, model.unpatchify, model.patchify)
                corr_dict[mask_ratio].append(corr.item())
                xcorr_dict[mask_ratio].append(xcorr.item())
                dtw_dict[mask_ratio].append(dtw.mean().item())

                # display reconstructions for the first batch only
                if data_iter_step == 0:
                    display_reconstruction_MAE1d(samples, outputs, masks, patch_size, relative_loss_channels, model.unpatchify, task="random")

                # save mse and corr of the shared 10% masked patches across all masking_ratios
                _, mse10, _, _, _, _, _ = model.forward_loss(samples, outputs, binary_masks[0.1].repeat(samples.shape[0], 1), padding_mask)
                mse10_dict[mask_ratio].append(mse10.mean().item())
                corr, xcorr, dtw = pcc_on_masked_patches(outputs, target, binary_masks[0.1].repeat(samples.shape[0], 1),
                           relative_loss_channels, model.unpatchify, model.patchify)
                corr10_dict[mask_ratio].append(corr.item())
                xcorr10_dict[mask_ratio].append(xcorr.item())
                dtw10_dict[mask_ratio].append(dtw.mean().item())  # change to be only on the shared 10%


    # Log epoch-level reconstruction performance
    for mask_ratio in mask_ratios_for_evaluation:
        wandb.log({f"Eval mean MSE per epoch (mask_ratio={mask_ratio})": np.mean(mse_dict[mask_ratio])})
        wandb.log({f"Eval s.d. MSE per epoch (mask_ratio={mask_ratio})": np.std(mse_dict[mask_ratio])})
        wandb.log({f"Eval mean Pearson corr per epoch (mask_ratio={mask_ratio})": np.mean(corr_dict[mask_ratio])})
        wandb.log({f"Eval s.d. Pearson corr per epoch (mask_ratio={mask_ratio})": np.std(corr_dict[mask_ratio])})
        wandb.log({f"Eval mean x-corr per epoch (mask_ratio={mask_ratio})": np.mean(xcorr_dict[mask_ratio])})
        wandb.log({f"Eval s.d. x-corr per epoch (mask_ratio={mask_ratio})": np.std(xcorr_dict[mask_ratio])})
        wandb.log({f"Eval mean DTW per epoch (mask_ratio={mask_ratio})": np.mean(dtw_dict[mask_ratio])})
        wandb.log({f"Eval s.d. DTW per epoch (mask_ratio={mask_ratio})": np.std(dtw_dict[mask_ratio])})

        wandb.log({f"Eval mean MSE10 per epoch (mask_ratio={mask_ratio})": np.mean(mse10_dict[mask_ratio])})
        wandb.log({f"Eval s.d. MSE10 per epoch (mask_ratio={mask_ratio})": np.std(mse10_dict[mask_ratio])})
        wandb.log({f"Eval mean Pearson corr10 per epoch (mask_ratio={mask_ratio})": np.mean(corr10_dict[mask_ratio])})
        wandb.log({f"Eval s.d. Pearson corr10 per epoch (mask_ratio={mask_ratio})": np.std(corr10_dict[mask_ratio])})
        wandb.log({f"Eval mean xcorr10 per epoch (mask_ratio={mask_ratio})": np.mean(xcorr10_dict[mask_ratio])})
        wandb.log({f"Eval s.d. xcorr10 per epoch (mask_ratio={mask_ratio})": np.std(xcorr10_dict[mask_ratio])})
        wandb.log({f"Eval mean DTW10 per epoch (mask_ratio={mask_ratio})": np.mean(dtw10_dict[mask_ratio])})
        wandb.log({f"Eval s.d. DTW10 per epoch (mask_ratio={mask_ratio})": np.std(dtw10_dict[mask_ratio])})


        # Print the correlation of the shared 10% masked patches across all masking_ratios
        print(f'Evaluation with masking ratio = {100*mask_ratio}%')
        print(f"Eval MSE averaged per epoch: "
              f"{np.mean(mse_dict[mask_ratio]):.2}  ({np.std(mse_dict[mask_ratio]):.2})")
        print(f"Eval MSE averaged per epoch on the shared 10% masked patches: "
              f"{np.mean(mse10_dict[mask_ratio]):.2}  ({np.std(mse10_dict[mask_ratio]):.2})")


    pass


def temporal_extrapolation(model,
                           dataloader,
                           device,
                           input_channels,
                           patch_size,
                           loss_channels):
    """Loop over batches and evaluate model for forecasting tasks at different time horizons.
    Log the mean and s.d. of the MSE and Pearson correlation for each time horizon.
    Args:
        model: The model to evaluate
        dataloader: The dataloader to loop over
        device: The device to run the model on
        input_channels: The input channels of the model
        patch_size: The patch size of the model
        loss_channels: The channels to evaluate the loss
    """
    # Initializations
    model.eval()
    mask_ratio = 0.3
    time_window_to_forcast = 1  # sec
    time_horizons = sorted([0, 10*60])  #sorted([0, 60, 5*60, 10*60, 60*60])  # sec
    sampling_rate = int(wandb.run.config['data_path'].split("SR_")[1].split("Hz")[0])
    relative_loss_channels = [input_channels.index(ch) for ch in loss_channels]
    mse_dict = {}
    corr_dict = {}
    xcorr_dict = {}
    dtw_dict = {}

    # extract first sample patches from dataloader
    with torch.no_grad():
        x = next(iter(dataloader))[0].to(device, non_blocking=True)
        x = model.patch_embed(x)  # (B, num_patches, embed_dim)

    mask_dir = os.path.join(PARENT_DIR, f'masks_for_{x.shape[1]}_patches')

    if not os.path.exists(os.path.join(mask_dir, f"forecast_mask_{mask_ratio}.bit")):
        # Generate temporal mask for forecasting task with specified masking ratio
        mkdirifnotexists(mask_dir)
        mask = temporal_masking(x, mask_ratio)[0]
        save_binary_vector(mask, os.path.join(mask_dir, f"forecast_mask_{mask_ratio}.bit"))
    else:
        # Load existing mask, ids_restore and ids_keep
        mask = read_binary_vector(
            os.path.join(mask_dir, f"forecast_mask_{mask_ratio}.bit"),
            x.shape[1]
        )

    # create corresponding ids_restore and keep:
    len_keep = int(x.shape[1] * (1 - mask_ratio))
    ids_restore = torch.arange(x.shape[1], device=device)
    ids_keep = torch.arange(len_keep, device=device)
    mask = torch.tensor(mask).to(device, non_blocking=True)

    # Initialize dictionaries for metrics
    for time_horizon in time_horizons:
        mse_dict[time_horizon] = []
        corr_dict[time_horizon] = []
        xcorr_dict[time_horizon] = []
        dtw_dict[time_horizon] = []

    # Evaluation on given dataset
    for data_iter_step, (samples, _ids) in enumerate(dataloader):
        with torch.no_grad():
            # compute validation loss (MSE) and pearson correlation for different time horizons:
            for time_horizon in time_horizons:
                # move samples to device and normalize
                samples = samples.to(device, non_blocking=True)

                # forward pass
                with torch.cuda.amp.autocast():
                    _, outputs, masks, padding_mask, _, target, _, _, _, _, _ = model(samples,
                                                                       mask=mask.repeat(samples.shape[0], 1),
                                                                       ids_restore=ids_restore.repeat(samples.shape[0], 1),
                                                                       ids_keep=ids_keep.repeat(samples.shape[0], 1))

                # extract the target and predictions for the time horizon in loss_channels
                B, C, L = samples.shape
                mask_bool = masks.bool()  # change mask into boolean vector
                masked_outputs = torch.stack(
                    [outputs[i, mask_bool[i], :] for i in range(B)])  # shape: (B, num_masked_patches, patch_seq_len*channels)
                masked_target = torch.stack([target[i, mask_bool[i], :] for i in
                                             range(B)])  # shape: (B, num_masked_patches, patch_seq_len*channels)

                outputs_unpatchified = model.unpatchify(masked_outputs)  # shape: (B, in_ch, seq_len)
                target_unpatchified = model.unpatchify(masked_target)  # shape: (B, in_ch, seq_len)
                window_to_evaluate = list(
                    range(sampling_rate * time_horizon, sampling_rate * (time_horizon + time_window_to_forcast)))

                # check that the time horizon is within the sequence length:
                if outputs_unpatchified.shape[2] < window_to_evaluate[-1]:
                    continue

                # display reconstructions for the first batch only
                if data_iter_step == 0:
                    display_reconstruction_MAE1d(samples, outputs, masks, patch_size, relative_loss_channels, model.unpatchify, "forecasting")


                if outputs_unpatchified.shape[1] == 1:
                    relative_loss_channels = [0]
                    C = 1

                outputs_to_evaluate = outputs_unpatchified[:, relative_loss_channels, window_to_evaluate]
                targets = target_unpatchified[:, relative_loss_channels, window_to_evaluate]

                # Check if the window_to_evaluate falls in the padding area
                cancel_padding = torch.ones(B, len(relative_loss_channels), 1, device=device)  # shape: (B, out_ch, 1)
                patch_size = int(masked_outputs.shape[2] / C)
                for i in range(B):
                    tmp = padding_mask.unsqueeze(-1).repeat(1, 1, patch_size)
                    tmp = model.unpatchify(tmp[i, :, :].unsqueeze(0))
                    if (tmp[:, :, window_to_evaluate] == 1).any():
                        cancel_padding[i, :, 0] = 0

                # save batch-level metrics:
                # mse
                mse = ((outputs_to_evaluate - targets) ** 2).mean(dim=1)[cancel_padding.squeeze().bool()]
                mse_dict[time_horizon].append(mse.mean().item())
                # Pearson corr
                corr = pcc(outputs_to_evaluate, targets).abs()[cancel_padding.squeeze().bool()]
                corr_dict[time_horizon].append(corr.mean().item())
                # cross-correlation
                xcorr = max_cross_correlation_fft(outputs_to_evaluate, targets)[cancel_padding.squeeze().bool()]
                xcorr_dict[time_horizon].append(xcorr.mean().item())
                # Dynamic Time Warping (DTW)
                dtw_loss = SoftDTWLossPyTorch(gamma=0.1, normalize=True)
                dtw = dtw_loss(outputs_to_evaluate.unsqueeze(1).float(), targets.unsqueeze(1).float())
                dtw_dict[time_horizon].append(dtw.mean().item())


    # Log epoch-level reconstruction performance
    for time_horizon in time_horizons:
        print(f'Evaluation of {time_window_to_forcast}sec in {time_horizon}sec time horizon (masking ratio = {100*mask_ratio}%):')
        print(f"MSE averaged per epoch: "
              f"{np.mean(mse_dict[time_horizon]):.2}  ({np.std(mse_dict[time_horizon]):.2})")
        print(f"Val Pearson corr averaged per epoch: "
              f"{np.mean(corr_dict[time_horizon]):.2}  ({np.std(corr_dict[time_horizon]):.2})")
        # log to wandb
        wandb.log({f"Val mean MSE per epoch (time horizon={time_horizon}sec)": np.mean(mse_dict[time_horizon])})
        wandb.log({f"Val s.d. MSE per epoch (time horizon={time_horizon}sec)": np.std(mse_dict[time_horizon])})
        wandb.log({f"Val mean Pearson corr per epoch (time horizon={time_horizon}sec)": np.mean(corr_dict[time_horizon])})
        wandb.log({f"Val s.d. Pearson corr per epoch (time horizon={time_horizon}sec)": np.std(corr_dict[time_horizon])})
        wandb.log({f"Val mean xcorr per epoch (time horizon={time_horizon}sec)": np.mean(xcorr_dict[time_horizon])})
        wandb.log({f"Val s.d. xcorr per epoch (time horizon={time_horizon}sec)": np.std(xcorr_dict[time_horizon])})
        wandb.log({f"Val mean DTW per epoch (time horizon={time_horizon}sec)": np.mean(dtw_dict[time_horizon])})
        wandb.log({f"Val s.d. DTW per epoch (time horizon={time_horizon}sec)": np.std(dtw_dict[time_horizon])})

    pass


def temporal_interpolation(model,
                           dataloader,
                           device,
                           input_channels,
                           patch_size,
                           loss_channels):
    """Loop over batches and evaluate model for imputation task of different time windows.
    Log the mean and s.d. of the MSE and Pearson correlation for each time window.
    Args:
        model: The model to evaluate
        dataloader: The dataloader to loop over
        device: The device to run the model on
        input_channels: The input channels of the model
        patch_size: The patch size of the model
        loss_channels: The channels to evaluate the loss
    """
    # Initializations
    model.eval()
    num_masked_patches = sorted([1, 10])  # num_masked_patches = sorted([1, 2, 5, 10])
    # sampling_rate = int(wandb.run.config['data_path'].split("SR_")[1].split("Hz")[0])
    relative_loss_channels = [input_channels.index(ch) for ch in loss_channels]
    mse_dict = {}
    corr_dict = {}
    xcorr_dict = {}

    # extract first sample patches from dataloader
    with torch.no_grad():
        x = next(iter(dataloader))[0].to(device, non_blocking=True)
        x = model.patch_embed(x)  # (B, num_patches, embed_dim)

    # define mask:
    start_mask = int(x.shape[1] / 2)

    # Initialize dictionaries for metrics
    for mask_len in num_masked_patches:
        mse_dict[mask_len] = []
        corr_dict[mask_len] = []
        xcorr_dict[mask_len] = []

    # Evaluation on given dataset
    for data_iter_step, (samples, _ids) in enumerate(dataloader):
        with torch.no_grad():
            # compute validation loss (MSE) and pearson correlation for different number of masked patches:
            for mask_len in num_masked_patches:
                # move samples to device and normalize
                samples = samples.to(device, non_blocking=True)
                B, C, L = samples.shape

                # Define mask
                mask = torch.zeros([x.shape[0], x.shape[1]], device=x.device)
                mask[:, start_mask:start_mask + mask_len] = 1
                ids_restore = torch.arange(mask.shape[1], device=device)
                ids_keep = ids_restore[mask[0] == 0]

                # forward pass
                with torch.cuda.amp.autocast():
                    mse, outputs, _, padding_mask,  _, target, _, _, _, _, _ = model(samples,
                                                         mask=mask,
                                                         ids_restore=ids_restore.repeat(B, 1),
                                                         ids_keep=ids_keep.repeat(B, 1))

                # save batch-level metrics:
                # mse
                mse_dict[mask_len].append(mse.mean().item())
                # Pearson cor
                mask = mask * (1 - padding_mask)
                corr, xcorr, _ = pcc_on_masked_patches(outputs, target, mask,
                                                   relative_loss_channels, model.unpatchify, model.patchify)
                corr_dict[mask_len].append(corr.cpu().detach().numpy())
                xcorr_dict[mask_len].append(xcorr.cpu().detach().numpy())

                # # display reconstructions for the first batch only
                # if data_iter_step == 0:
                #     display_reconstruction_MAE1d(samples, outputs, masks, patch_size, relative_loss_channels, model.unpatchify, "interpolation")


    # Log epoch-level reconstruction performance
    for mask_len in num_masked_patches:
        print(f'Evaluation of imputation of {mask_len} masked patches in the middle on the sequence:')
        print(f"MSE averaged per epoch: "
              f"{np.mean(mse_dict[mask_len]):.2}  ({np.std(mse_dict[mask_len]):.2})")
        print(f"Val Pearson corr averaged per epoch: "
              f"{np.mean(corr_dict[mask_len]):.2}  ({np.std(corr_dict[mask_len]):.2})")
        # log to wandb
        wandb.log({f"Val mean MSE per epoch (num masked patches ={mask_len})": np.mean(mse_dict[mask_len])})
        wandb.log({f"Val s.d. MSE per epoch (num masked patches ={mask_len})": np.std(mse_dict[mask_len])})
        wandb.log({f"Val mean Pearson corr per epoch (num of masked patches ={mask_len})": np.mean(corr_dict[mask_len])})
        wandb.log({f"Val s.d. Pearson corr per epoch (num masked patches ={mask_len})": np.std(corr_dict[mask_len])})
        wandb.log({f"Val mean x-corr per epoch (num masked patches ={mask_len})": np.mean(xcorr_dict[mask_len])})
        wandb.log({f"Val s.d. x-corr per epoch (num masked patches ={mask_len})": np.std(xcorr_dict[mask_len])})
    pass


def save_binary_vector(binary_vector, filename):
    """
    Save a binary vector to a bit file.

    :param binary_vector: NumPy array of 0s and 1s
    :param filename: Path to the output bit file
    """
    # move binary_vector to cpu
    binary_vector = binary_vector.cpu().detach().numpy()
    # Ensure the input is a binary vector (0s and 1s)
    if not np.all(np.isin(binary_vector, [0, 1])):
        raise ValueError("Input must be a binary vector containing only 0s and 1s")

    # Pack bits into bytes
    packed_bits = np.packbits(binary_vector.astype(bool))

    # Save to file
    with open(filename, 'wb') as f:
        f.write(packed_bits.tobytes())


# Function to read back the binary vector (for demonstration)
def read_binary_vector(filename, original_length):
    """
    Read back a binary vector from a bit file.

    :param filename: Path to the bit file
    :param original_length: Length of the original binary vector
    :return: Numpy array of binary vector
    """
    with open(filename, 'rb') as f:
        packed_bits = np.frombuffer(f.read(), dtype=np.uint8)

    # Unpack bits
    unpacked_bits = np.unpackbits(packed_bits)

    # Trim to original length
    return unpacked_bits[:original_length]


def nested_random_masking(x, mask_ratios):
    """
    Perform nested random masking where higher percentage masks include all elements from lower percentage masks.

    Args:
        x: Input tensor of shape [N, L, D] where:
           - N is the batch size,
           - L is the sequence length,
           - D is the embedding dimension.
        mask_ratios: List of mask ratios in ascending order (e.g., [0.1, 0.3, 0.5, 0.7, 0.9])

    Returns:
        Dictionary containing for each mask ratio:
        - x_masked: Masked sequence
        - mask: Binary mask (1 for masked, 0 for kept)
        - ids_restore: Indices to restore original sequence order
    """
    N, L, D = x.shape  # batch, length, dim
    results = {}

    # Generate one random noise tensor for all masks
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is masked
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    for mask_ratio in mask_ratios:
        len_keep = int(L * (1 - mask_ratio))

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Generate binary mask: 0 is keep, 1 is mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        results[mask_ratio] = {
            'mask': mask,
            'ids_restore': ids_restore,
            'ids_keep': ids_keep,
        }

    return results


def temporal_masking(x, mask_ratio):
    """
    Perform masking according to the input mask_ratio, and sort the masked patches to be t the end of the timeseries
     (beginning unmasked), for forcasting task.
    Args:
        x: Input tensor of shape [N, L, D] where:
           - N is the batch size,
           - L is the sequence length,
           - D is the embedding dimension.
        mask_ratio: mask ratio to be used for masking
    Returns:
        - mask: Binary mask (1 for masked, 0 for kept)
    """
    N, L, D = x.shape  # batch, length, dim

    # Generate mask with mask_ratio of the patches to be masked at the end of the timeseries, other unmasked
    len_keep = int(L * (1 - mask_ratio))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    return mask


display_reconstruction_MAE1d_on_segments = display_reconstruction_MAE1d
display_reconstruction_MAE1d_on_night = display_reconstruction_MAE1d