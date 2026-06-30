# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
import json
import logging
import argparse
from datetime import timedelta
import time
from pathlib import Path
import timm
assert timm.__version__ == "0.5.4", f"Expected timm==0.5.4, got {timm.__version__}"
import timm.optim.optim_factory as optim_factory
from model.architecture.util import misc as misc
from model.architecture.util.misc import NativeScalerWithGradNormCount as NativeScaler
import model.architecture.mae_vit as models_mae
from model.training.util import lr_sched as lr_sched
try:
    from reptrix import rankme
except ImportError:
    rankme = None

# Imports from our files
from model.utils import *
import model.data.datasets as sleep_datasets
from downstream.reconstruction import *
# GPU performance: L40S-friendly settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.multiprocessing.set_start_method('spawn', force=True)

# OUR PARAMETERS:

# set dataset globals
RESAMPLE_RATE = 125  # Hz

# set hyperparameters for model
ARCHITECTURE = "MAE1d_on_segments"  # {MAE1d_on_night/MAE1d_on_segments/MAEspec}

if ARCHITECTURE == "MAE1d_on_night":  # whole night
    MODEL = "sleep_1d_original"
    N_ITEMS = None    # 100 for subset or None for full dataset
    ZSCORE = False  # True for zscored data, False for normalized data
    EMBEDED_DIM = 256
    MASK_PROB = 0.5
    LR = 0.1  # =base LR,  0.001 used in original Meta's MAE with AdamW optimizer
    if RESAMPLE_RATE == 10:  # 10Hz data
        BATCH_SIZE = 64
        INPUT_CHANNELS = [2]
        LOSS_CHANNELS = [2]
        INPUT_SIZE = 2 ** 18  # ~7.3h
        PATCH_SIZE = 1024  # ~1.7min
        PATH_FOR_PROCESSED_DATA = PROJECT_DIR + f"preprocessed_data_SEQ_{(INPUT_SIZE / RESAMPLE_RATE / 60 / 60):.2}h_SR_{RESAMPLE_RATE}Hz_full/"
    else:  # 125Hz Hz data with PAT preprocessed using pyPPG filters
        BATCH_SIZE = int(8*torch.cuda.device_count())
        INPUT_CHANNELS = [0,1,2] #,3,4,5,6]
        LOSS_CHANNELS = [2]
        INPUT_SIZE = 2 ** 21   # ~4.6h
        PATCH_SIZE = 2 ** 13  # 8192 = ~2.2min
        PATH_FOR_PROCESSED_DATA = PROJECT_DIR + f"preprocessed_data_SR_125Hz_gold"

elif ARCHITECTURE == "MAE1d_on_segments":
    MODEL = "sleep_1d_original"
    N_ITEMS = None   # 10,000 for subset or None for full dataset
    ZSCORE = True  # True for zscored data, False for normalized data
    SEGMENT_LENGTH = 120 # seconds  {10/30/60/120/240/480/960/1920/3840/7680}
    INPUT_SIZE = SEGMENT_LENGTH * 125  # n_sec * 125Hz  # effective segment length
    PATCH_SIZE = 125  # 1sec
    BATCH_SIZE = 128 #int(128*torch.cuda.device_count())
    INPUT_CHANNELS = [0,1,2] # PAT channel
    LOSS_CHANNELS = [2]
    LR = 0.001
    EMBEDED_DIM = 256
    MASK_PROB = 0.5
    PATH_FOR_PROCESSED_DATA = PROJECT_DIR + f"preprocessed_data_SR_125Hz_{SEGMENT_LENGTH}s_segments_gold__segmented"

else:
    MODEL = "sleep_spec_demo_1"
    INPUT_SIZE = (64, 4096)
    N_ITEMS = None
    PATCH_SIZE = (4, 64)
    BATCH_SIZE = 4
    INPUT_CHANNELS = [2]
    LOSS_CHANNELS = [2]
    PATH_FOR_PROCESSED_DATA = PROJECT_DIR + f"preprocessed_data_SEQ_7.3h_SR_{RESAMPLE_RATE}Hz_spec_win_12sec_overlap_0.5/"

LOSS_VERSION = "v3.1"        # 'v1.0' (MSE on all input channels), 'v2.0' (MSE+variance on all input channels),
                             # 'v3.1' (weighted MSE per channels) -> to select only loss_channels
                             # 'v3.3. DTW loss 
MULTI_LOSS = False
AUGMENTATIONS = False
EPOCHS = 400  # defaults to 400 in META's MAE
WARMUP_EPOCHS = 2   # defaults to 40 in META's MAE
WD = 0.05  # 0.05 used in original Meta's MAE with AdamW optimizer
SCHEDULER = 'None'  # if None use default scheduler from meta's framework, otherwise specify,
                    # e.g.:'ReduceLROnPlateau_MinTrainLoss', 'NoWD_NoScheduler'
PATIENCE = 3  # number of epochs to wait without improvement (for reduce on plateau)
NORM_PIX_LOSS = False
INCREASE_MASK_RATIO = False  # whether to increase the masking ratio in the second half of training
VARIOUS_MASKING_STRATEGIES = False  # whether to use three masking strategies (random, begining or end)
PULSE_SEGMENTATION = False
EVAL_MODEL = True
USE_CLS_TOKEN = True
POOLING_METHOD = 'max'  #'max' or 'avg' (not relevant if using cls token)

OVERFIT_EXP = False  # set to True for train debugging (skips eval)
MIN_EPOCHS_FOR_RESUME = 10
MAX_STEPS_PER_EPOCH = None
DATA_FORMAT = '.pt'  # '.pt' for torch tensor, '.npz' for compressed numpy array
DEBUG = False
TIME_BUDGET = 6 * 60 * 60  # [sec] - 6h for normal.q partition, 12h for long.q partition
NUM_WORKERS = 2 #int(2*torch.cuda.device_count()) if torch.cuda.is_available() else 0  # 0 for no multiprocessing
PREFETCH_FACTOR = None  # 2–4 typical for L40S; higher can overlap data load with compute
USE_AMP = False  # mixed precision (FP16) for L40S; set False if numerical issues
LOG_EVERY_N_STEPS = 10  # reduce GPU sync from .item()/wandb by logging every N steps
COLLECT_LATENT_EVERY_N_STEPS = 0  # 0 = disable per-step latent collection (avoids .cpu() sync every batch)
#########################################

def get_args_parser(run_config):
    # Generic parameters
    parser = argparse.ArgumentParser(description='MAE pre-training', add_help=False)
    parser.add_argument("--architecture", type=str, default=run_config["architecture"],
                        help="Model architecture: {MAEspec/MAE1d}")
    parser.add_argument('--batch_size', default=run_config["batch_size"], type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')  # old = 4
    parser.add_argument("--patch_size", type=int, default=run_config["patch_size"],
                        help="Size of the patch. int for 1d, int tuple for 2d. must be powers of 2.")
    parser.add_argument('--epochs', default=run_config["epochs"], type=int)  # old = 400
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model specific parameters
    parser.add_argument('--model', default=run_config["model"], type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=run_config["input_size"], type=int,
                        help="samples input size. int for 1d, tuple for 2d.")
    parser.add_argument('--n_items', type=int, default=run_config["n_items"])
    parser.add_argument('--emb_dim', default=run_config["emb_dim"], type=int, help='Embeddings dimension')
    parser.add_argument("--loss_channels", type=int, default=run_config["loss_channels"], help="List of loss channels.")
    parser.add_argument("--input_channels", type=list, default=run_config["input_channels"],
                        help="List of input channel indices")
    parser.add_argument("--n_channels", type=int, default=run_config["n_channels"],
                        help="number of input channel(s)")
    parser.add_argument("--multi_loss", type=bool, default=run_config["multi_loss"],
                        help="Whether to use or not multi-loss (adding supervision on selected features).")
    parser.add_argument("--loss_version", type=str, default=run_config["loss_version"],
                        help="Version of the loss function.")
    parser.add_argument('--norm_pix_loss', type=bool, default=run_config["norm_pix_loss"],
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--eval_model', type=bool, default=run_config["eval_model"],
                        help='Whether to eval the model - run eval epock after each train epoch.')
    parser.add_argument("--use_cls_token", type=bool, default=run_config['use_cls_token'],
                        help="Whether to use class token as latent representation or not.")
    parser.add_argument("--pooling_method", type=str, default=run_config['pooling_method'],
                        help="Specify the pooling method ('avg' or 'max') to use on latent representations over patches,"
                             " if not using class token.")
    parser.add_argument('--display_reconstruction', type=bool, default=run_config["display_reconstruction"],
                        help='Whether to display reconstructed outputs during training.')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--base_learning_rate', type=float, default=run_config["base_learning_rate"], metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=run_config["warmup_epochs"], metavar='N',
                        help='epochs to warmup LR')  # meta's default was set to 40
    parser.add_argument('--weight_decay', type=float, default=run_config["weight_decay"],
                        help='weight decay (default: 0.05)')
    parser.add_argument('--scheduler', type=str, default=run_config["scheduler"],
                        help="scheduler type: None (for Meta's scheduler) or ReduceLROnPlateau_MinTrainloss")
    parser.add_argument('--patience', type=int, default=run_config["patience"],
                        help="number of epochs to wait without improvement (relevant only for reduce on plateau scheduler)")

    # Dataset parameters
    parser.add_argument('--data_path', default=run_config["data_path"], type=str, help='dataset path')
    parser.add_argument('--output_dir', default=run_config["output_dir"], help='path where to save, empty for no saving')
    parser.add_argument('--zscore', default=run_config["zscore"], type=bool, help='True to apply zscore on input, False for dataset normalization')
    parser.add_argument('--save_every_epoch', default=1, type=int, help='save_every_epoch')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=SEED, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=run_config['num_workers'], type=int)  # set to 0 for no multiprocessing
    parser.add_argument('--prefetch_factor', default=run_config['prefetch_factor'], type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')  # NEED TO CHECK!
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')  # NEED TO CHECK!
    parser.set_defaults(
        pin_mem=True)  # NEED TO CHECK!- If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,  # NEED TO CHECK!
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)  # NEED TO CHECK!
    parser.add_argument('--dist_on_itp', action='store_true')  # NEED TO CHECK!
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')  # NEED TO CHECK!

    # args for "MAE1d":
    parser.add_argument('--mask_ratio', default=run_config['mask_ratio'], type=float,
                        help='mask ratio (percentage of removed patches)')
    parser.add_argument('--increase_mask_ratio', type=bool, default=run_config['increase_mask_ratio'])
    parser.add_argument('--various_masking_strategies', type=bool, default=run_config['various_masking_strategies'])
    parser.add_argument('--augmentations', type=bool, default=run_config['augmentations'])
    # args for "MAE2d" architecture:
    parser.add_argument('--audio_exp', type=bool, default=False, help='audio exp')  # OLD = TRUE
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)  # pretraining 0
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)  # pretraining 0
    parser.add_argument('--mask_t_prob', default=0.7, type=float, help='ratio of masking time')
    parser.add_argument('--mask_f_prob', default=0.3, type=float, help='ratio of masking freq')
    parser.add_argument('--mask_2d', type=bool, default=False, help='use 2d masking')

    # other arguments inherited from Meta's code
    parser.add_argument("--mixup", type=float, default=0,
                        help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.0, help="contrastive loss weight")
    parser.add_argument("--omega", type=float, default=1.0, help="reconstruction loss weight")
    parser.add_argument('--mode', default=0, type=int, help='contrastive mode')
    parser.add_argument('--use_custom_patch', type=bool, default=False,
                        help='use custom patch and override timm PatchEmbed')
    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument('--roll_mag_aug', type=bool, default=False, help='use roll_mag_aug')
    parser.add_argument('--split_pos', type=bool, default=False, help='use splitted pos emb')
    parser.add_argument('--pos_trainable', type=bool, default=False, help='use trainable pos emb')
    parser.add_argument('--use_nce', type=bool, default=False, help='use use_nce')
    parser.add_argument('--decoder_mode', default=1, type=int, help='decoder mode 0: global attn 1: swined local attn')
    parser.add_argument('--init_audio_with_video_mae', type=bool, default=False, help='init_audio_with_video_mae')
    parser.add_argument('--no_shift', type=bool, default=False, help='no_shift')
    parser.add_argument('--use_amp', type=bool, default=run_config.get('use_amp', True),
                        help='Use automatic mixed precision (FP16) for faster training on L40S etc.')
    parser.add_argument('--log_every_n_steps', type=int, default=run_config.get('log_every_n_steps', 10),
                        help='Log batch metrics to wandb every N steps (reduces GPU sync).')
    parser.add_argument('--collect_latent_every_n_steps', type=int, default=run_config.get('collect_latent_every_n_steps', 0),
                        help='Collect latent dataframe every N steps; 0 = never (avoids .cpu() sync).')
    return parser


def save_entire_dataset(dataset, out_path, desc="Saving"):
    all_data = []
    all_recordings = []

    if DATA_FORMAT == '.pt':     # save as pt file
        for data, recording in tqdm(dataset, desc=desc):
            all_data.append(data)  # tensor
            all_recordings.append(recording)  # string or path
        all_data = torch.utils.data.dataloader.default_collate(all_data)
        torch.save({'data': all_data, 'recordings': all_recordings}, out_path+DATA_FORMAT)

    else:      # save as compressed npy
        for data, recording in tqdm(dataset, desc=desc):
            all_data.append(data.numpy())
            all_recordings.append(recording)  # string or path
        all_data = np.stack(all_data, axis=0)
        all_recordings = np.array(all_recordings)  # convert to numpy array
        np.savez_compressed(out_path+DATA_FORMAT, data=all_data, recordings=all_recordings)
    print(f"Saved {len(dataset)} items to {out_path+DATA_FORMAT}")

class LoadedTensorNameDataset(Dataset):
    def __init__(self, path):
        if DATA_FORMAT == '.pt':
            loaded = torch.load(path, map_location='cpu', weights_only=True)  # ,mmap=True)
            self.data = loaded['data']  # list of tensors
            self.names = loaded['recordings']  # list of strings
        else:  # .npz format
            loaded = np.load(path, allow_pickle=True)  # , mmap_mode='r')
            self.data = torch.from_numpy(loaded['data'])  # list of tensors
            self.names = loaded['recordings'].tolist()  # list of strings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.names[idx]


def main(args):
    print('======================= starting pretrain =======================')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # program init
    device = torch.device(args.device)
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    setup_seed(args.seed)
    torch.cuda.empty_cache()
    # 'high' = use TF32 on Ampere+ (e.g. L40S) for faster matmul; 'medium' is more conservative
    torch.set_float32_matmul_precision('medium')

    # check if a folder with preprocessed data already exists, and if not crate one:
    assert os.path.exists(args.data_path), (f"{args.data_path} could not be found."
                                            f"\n Please run timeseries_preprocessing.py with these parameters to create one.")

    # assert loss channels contained in input channels
    assert all(ch in args.input_channels for ch in args.loss_channels), \
        "all loss channels must be contained in the input channels. input channels: {}, loss channels: {}".format(
            args.input_channels, args.loss_channels)

    ######################################
    # Datasets init
    ######################################

    # Define transforms for training (includes normalization & augmentations)
    if args.zscore:  # perform temporal z-score normalization per sample
        test_transform = transforms.Compose([TemporalChannelwiseZScore(channels=args.input_channels)])
        if args.augmentations:
            train_transform = transforms.Compose([
                TemporalChannelwiseZScore(channels=args.input_channels),  # temporal z-score
                # Add time-series augmentations here -- TO BE CHECKED how behave with 0-6 channels
                Jitter(std=0.01),  # Add Gaussian noise
                RandomScaling(scale_range=(0.9, 1.1)),  # Scale amplitude
                RandomTimeMask(mask_ratio=0.05),  # Mask out 5% of the time series
                RandomChannelDropout(drop_prob=0.1),  # Drop random channels
            ])
        else:
            train_transform = transforms.Compose([
                TemporalChannelwiseZScore(channels=args.input_channels),  # temporal z-score
            ])
        # define length by: segments_length if ARCHITECTURE == "MAE1d_on_segments" else input_size
        length = f'{int(args.input_size / RESAMPLE_RATE)}s' if args.architecture == "MAE1d_on_segments" else f'{args.input_size / RESAMPLE_RATE / 60 / 60}h'
        transformed_train_dataset_path = PROJECT_DIR + f'gold_subsets/zscored_train_{len(args.input_channels)}ch_{length}_{args.architecture.split("_")[2]}_N{run_config["n_items"]}'
        transformed_val_dataset_path = PROJECT_DIR + f'gold_subsets/zscored_val_{len(args.input_channels)}ch_{length}_{args.architecture.split("_")[2]}_N{run_config["n_items"]}'

    else:  # perform dataset normalization , e.g. if running on whole night with multiple channels
        if 0:
            # Define Normalization and transformations
            print("Define Normalization and transformations")
            # Compute mean and std from the training dataset
            train_data = MAE1d(directory=args.data_path,
                               split_dir='randomstate1',
                               architecture=args.architecture,
                               channels=args.input_channels,
                               split="gold_train",
                               compute_stats=True)
            train_mean, train_std = train_data.mean, train_data.std  # Save computed stats
            print('Mean and s.d. for normalization:', train_mean, train_std)
        else:
            if RESAMPLE_RATE == 10:
                # use predefined mean and std, computed on 1000 nights with original preproessing (10Hz downsampling)
                train_mean = torch.tensor(
                    [[8.6755e+01], [5.5093e+01], [2.3232e+03], [1.7623e+03], [1.8673e+01], [1.8239e+00],
                     [3.5644e+01]])
                train_std = torch.tensor(
                    [[2.7840e+01], [1.9883e+01], [8.0125e+02], [6.3571e+02], [1.2794e+03], [1.2164e+00],
                     [1.1477e+01]])
            else:
                # 125Hz data
                train_mean = torch.tensor(
                    [[96.1361], [67.0277], [25.7122], [1944.5280], [-11.5019], [2.1329], [40.2803]])
                train_std = torch.tensor(
                    [[2.5843e+00], [1.3267e+01], [1.9327e+02], [3.5462e+02], [5.0193e+03], [1.4147e+00],
                     [3.9806e+00]])

        # Define transforms for validation/test (only normalization)
        test_transform = transforms.Compose([
            TimeSeriesTransform(train_mean, train_std, channels=args.input_channels)
            # Use training stats for normalization
        ])

        # Define transforms for training (includes normalization & augmentations)
        if args.augmentations:
            train_transform = transforms.Compose([
                TimeSeriesTransform(train_mean, train_std, channels=args.input_channels),  # Standardize
                # Add time-series augmentations here if needed -- TO BE CHECKED how behave with 0-6 channels
                Jitter(std=0.01),  # Add Gaussian noise
                RandomScaling(scale_range=(0.9, 1.1)),  # Scale amplitude
                RandomTimeMask(mask_ratio=0.05),  # Mask out 5% of the time series
                RandomChannelDropout(drop_prob=0.1),  # Drop random channels
            ])
        else:
            train_transform = transforms.Compose([
                TimeSeriesTransform(train_mean, train_std, channels=args.input_channels),  # Standardize
            ])
        length = f'{int(args.input_size / RESAMPLE_RATE)}s' if args.architecture == "MAE1d_on_segments" else f'{args.input_size / RESAMPLE_RATE / 60 / 60}h'
        transformed_train_dataset_path = PROJECT_DIR + f'gold_subsets/normalized_train_{len(args.input_channels)}ch_{length}_{args.architecture.split("_")[2]}_N{run_config["n_items"]}'
        transformed_val_dataset_path = PROJECT_DIR + f'gold_subsets/normalized_val_{len(args.input_channels)}ch_{length}_{args.architecture.split("_")[2]}_N{run_config["n_items"]}'

    # Load preprocessed dataset (subset) if exists in storage or create and save it:
    if (os.path.exists(transformed_train_dataset_path + DATA_FORMAT) and
            os.path.exists(transformed_val_dataset_path + DATA_FORMAT) and
            (run_config["n_items"] is not None)):
        print(f"Preprocessed subset dataset exists, loading data from '{transformed_val_dataset_path}'...")

        train_data = LoadedTensorNameDataset(transformed_train_dataset_path + DATA_FORMAT)
        val_data = LoadedTensorNameDataset(transformed_val_dataset_path + DATA_FORMAT)
        wandb.log({"Train N": len(train_data)})
        wandb.log({"Validation N": len(val_data)})

    else:
        print("Load train dataset")
        train_data = sleep_datasets.__dict__[args.architecture + "_dataset"](directory=args.data_path,
                                                                             split_dir='randomstate1',
                                                                             architecture=args.architecture,
                                                                             channels=args.input_channels,
                                                                             split="gold_train",
                                                                             transform=train_transform,
                                                                             n_items=run_config["n_items"],
                                                                             segment_length=args.input_size)
        wandb.log({"Train N": len(train_data)})

        print("Load Validation dataset")
        val_data = sleep_datasets.__dict__[args.architecture + "_dataset"](directory=args.data_path,
                                                                           split_dir='randomstate1',
                                                                           architecture=args.architecture,
                                                                           channels=args.input_channels,
                                                                           split="gold_val",
                                                                           transform=test_transform,
                                                                           n_items=run_config["n_items"],
                                                                           segment_length=args.input_size)
        wandb.log({"Validation N": len(val_data)})
        if run_config["n_items"] is not None:
            save_entire_dataset(train_data, out_path=transformed_train_dataset_path, desc="Saving Train Data")
            save_entire_dataset(val_data, out_path=transformed_val_dataset_path, desc="Saving Validation Data")
            return

    # load dataset for downstream tasks:
    if 'night' in args.architecture:
        ds_data = sleep_datasets.__dict__[args.architecture + "_dataset"](directory=args.data_path,
                                                                      split_dir='randomstate1',
                                                                      architecture=args.architecture,
                                                                      channels=args.input_channels,
                                                                      split="gold_all",
                                                                      transform=test_transform,
                                                                      n_items=600, # None for all train and val data
                                                                      segment_length=args.input_size)
        val_data = LoadedTensorNameDataset(transformed_val_dataset_path.replace('None', '100') + DATA_FORMAT)
    elif args.input_size > 15000:
        ds_data = sleep_datasets.__dict__[args.architecture + "_dataset"](directory=args.data_path,
                                                                      split_dir='randomstate1',
                                                                      architecture=args.architecture,
                                                                      channels=args.input_channels,
                                                                      split="gold_all",
                                                                      transform=test_transform,
                                                                      n_items=600, # None for all train and val data
                                                                      segment_length=args.input_size)
        val_data = ds_data
    else:  # if segment take the saved subset of 10K segments
        ds_data = LoadedTensorNameDataset(transformed_val_dataset_path.replace('None', '10000') + DATA_FORMAT)
        val_data = LoadedTensorNameDataset(transformed_val_dataset_path.replace('None', '10000') + DATA_FORMAT)

    wandb.log({"latent N": len(ds_data)})

    ######################################
    # Dataloader init
    ######################################

    if args.distributed:  # args.distributed: # old = true, not distributed GPU
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(train_data, num_replicas=num_tasks, rank=global_rank,
                                                            shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        sampler_eval = torch.utils.data.DistributedSampler(val_data, num_replicas=num_tasks, rank=global_rank,
            shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_data) if not OVERFIT_EXP else None
        sampler_eval = torch.utils.data.RandomSampler(val_data)

    global_rank = 1
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)  # log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = SafeDataLoaderWrapper(torch.utils.data.DataLoader(train_data, sampler=sampler_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    prefetch_factor=args.prefetch_factor,
                                                    collate_fn=collate_fn_skip_anomalies if args.architecture == "MAE1d_on_segments" else safe_collate,
                                                    pin_memory=args.pin_mem,  # maybe need to change to false
                                                    persistent_workers=True,
                                                    drop_last=True, ))

    data_loader_eval = SafeDataLoaderWrapper(torch.utils.data.DataLoader(val_data, sampler=sampler_eval,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   prefetch_factor=args.prefetch_factor,
                                                   collate_fn=collate_fn_skip_anomalies if args.architecture == "MAE1d_on_segments" else safe_collate,
                                                   pin_memory=args.pin_mem,
                                                   persistent_workers=True,
                                                   drop_last=True, ))

    data_loader_ds = SafeDataLoaderWrapper(torch.utils.data.DataLoader(ds_data, sampler=None,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 prefetch_factor=args.prefetch_factor,
                                                 collate_fn=collate_fn_skip_anomalies if args.architecture == "MAE1d_on_segments" else safe_collate,
                                                 pin_memory=args.pin_mem,
                                                 persistent_workers=True,
                                                 drop_last=True, ))

    ######################################
    # Model init
    ######################################

    relative_loss_channels = [args.input_channels.index(ch) for ch in args.loss_channels]
    assert len(args.input_channels) == args.n_channels, (f"Number of input channels ({args.n_channels}) "
                                                         f"must match the length of input channels list ({args.input_channels})")
    model = models_mae.__dict__[args.model](in_chans=len(args.input_channels),
                                            input_size=args.input_size,
                                            patch_size=args.patch_size,
                                            embed_dim=args.emb_dim,
                                            decoder_embed_dim=args.emb_dim,
                                            norm_pix_loss=args.norm_pix_loss,
                                            loss_version=args.loss_version,
                                            loss_channels=relative_loss_channels,
                                            ch_weights=None,
                                            )
    # print and lof num of params
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({f"parameters": total_params})
    print(f"Total number of parameters: {total_params}")

    # model = model.to(torch.bfloat16)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.base_learning_rate * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        print('use distributed!!')
        misc.init_distributed_mode(args)
        device = global_rank % torch.cuda.device_count()
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        model_without_ddp = model.module


    # Define optimizer and scheduler
    if args.scheduler == 'ReduceLROnPlateau_MinTrainLoss':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)
    elif args.scheduler == 'NoWD_NoScheduler':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = 'NoWD_NoScheduler'
    else:  # optimizer and scheduler used in vanilla MAE
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        scheduler = None
    print(optimizer)
    loss_scaler = NativeScaler() if getattr(args, 'use_amp', USE_AMP) else None

    # Load model for continued training if a checkpoint is saved for this run.id
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    ######################################
    # model training
    ######################################

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_losses = []
    val_loss = []
    best_val_loss = float("inf")
    last_epoch_duration = None
    for epoch in range(args.start_epoch, args.epochs):

        # # Check if there's enough time for another epoch
        # if last_epoch_duration is not None:
        #     elapsed_time = time.time() - start_time
        #     remaining_time = TIME_BUDGET - elapsed_time
        #     if remaining_time < last_epoch_duration:
        #         print(f"Not enough time for another epoch. Stopping at epoch {epoch}.")
        #         wandb.finish()
        #         # generate an ERROR
        #         sys.exit(1)
        epoch_start_time = time.time()

        # Train
        print(f'Train Epoch [{epoch + 1}/{str(args.epochs)}]')
        wandb.log({"epoch": epoch + 1})
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model_without_ddp, data_loader_train,
                                      optimizer, device, epoch, loss_scaler,
                                      log_writer=log_writer,
                                      args=args,
                                      scheduler=scheduler,
                                      display_reconstruction=False,
                                      cal_pearson=False)  # returns (lr, loss, corr) dictionary
        train_losses.append(train_stats["loss"])

        if args.eval_model:
            ######################################
            # model evaluation on validation set
            ######################################
            # Eval Validation reconstruction tasks
            print(f'Eval Validation Epoch [{epoch + 1}/{str(args.epochs)}]')
            with torch.no_grad():
                val_stats = eval_one_epoch(model_without_ddp, data_loader_eval, device, epoch, log_writer=log_writer, args=args,
                    display_reconstruction=False, cal_pearson=True)
            current_val_loss = val_stats["loss"]
            val_loss.append(current_val_loss)

            # # Eval latent representations quality
            # latent_df = get_latent_features(model=model_without_ddp, data_loader=data_loader_ds, device=device,
            #                                       use_cls_token=args.use_cls_token, pooling_method=args.pooling_method)
            # score = rankme.get_rankme(
            #     torch.tensor(latent_df.drop('Recordings', axis=1).values, dtype=torch.float32).to(device))
            # wandb.log({f"RankMe score": score})
            #
            # # Downstream predictions
            # rs = 1
            # scores = linear_probing(features_df_list=[latent_df],
            #                features_names_list=["ssl_latent"],
            #                labels_df=get_df(GOLD_RECORDS_PATH)[
            #                    ["Recordings", "age", "gender", "bmi", "lying_blood_pressure_systolic"]],
            #                train_with_cross_val=True,
            #                cv=KFold(n_splits=5, shuffle=True, random_state=rs),
            #                log=True,
            #                display=True,
            #                targets=["age"])
            # wandb.log({f'Pearson_corr_{scores.target_name.iloc[i]}_rs{rs}': scores.eval_scores.iloc[i]["corr"] for i in
            #            range(len(scores))})
            # wandb.log(
            #     {f'Pearson_corr_pval_{scores.target_name.iloc[i]}_rs{rs}': scores.eval_scores.iloc[i]["corr_pval"] for i
            #      in range(len(scores))})

        ######################################
        # Save model
        if args.output_dir and args.epochs > MIN_EPOCHS_FOR_RESUME and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, f"log_{wandb.run.id}.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        wandb.log({"current_val_loss": current_val_loss})

        # # Early stopping and best model tracking
        # if current_val_loss < best_val_loss:
        #     best_val_loss = current_val_loss
        #
        #     # Save the best model
        #     if args.output_dir and misc.is_main_process():
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'val_loss': best_val_loss,
        #         }, os.path.join(args.output_dir, f"best_model_{wandb.run.id}.pth"))
        #
        #     wandb.log({"best_val_loss": best_val_loss})

        # Track how long the epoch took
        epoch_end_time = time.time()
        last_epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch} completed in {last_epoch_duration:.2f} seconds.")
        # end of epoch

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    ######################################
    # Post-training evaluation
    ######################################

    # Load the best model saved
    if args.output_dir and misc.is_main_process():
        best_model_path = os.path.join(args.output_dir, f"best_model_{wandb.run.id}.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
            model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['val_loss']
            print(f"Best validation loss: {best_val_loss}")
    else:
        print("No best model found, using the last saved model.")

    # check reconstructed signal on first evaluation batch
    function_name = f"display_reconstruction_{args.architecture}"
    reconstruction_func = globals().get(function_name)
    samples, _ = next(iter(data_loader_eval))
    if isinstance(samples, list):
        samples = torch.stack(samples, dim=0)
    samples = samples.to(device, non_blocking=True)
    with torch.no_grad():
        _, outputs, masks, padding_mask, _, _, _, _, _, _, _ = model(samples, mask_ratio=args.mask_ratio)
    reconstruction_func(samples, outputs, masks, args.patch_size, relative_loss_channels, model_without_ddp.unpatchify, "training_task_valset")

    random_masking_reconstruction(model_without_ddp, data_loader_eval, device, args.input_channels, args.patch_size, args.loss_channels)

    temporal_extrapolation(model, data_loader_eval, device, args.input_channels, args.patch_size, args.loss_channels)

    # temporal_interpolation(model, data_loader_eval, device, args.input_channels, args.patch_size, args.loss_channels)

    # Eval all (train + val) latent representations quality (rankme)
    latent_df = get_latent_features(model=model, data_loader=data_loader_ds, device=device,
                                    use_cls_token=args.use_cls_token, pooling_method=args.pooling_method)
    score = rankme.get_rankme(
        torch.tensor(latent_df.drop('Recordings', axis=1).values, dtype=torch.float32).to(device))
    wandb.log({f"RankMe score": score})

    # Downstream predictions
    linear_probing(features_df_list=[latent_df],
                   features_names_list=["ssl_latent"],
                   labels_df=get_df(GOLD_RECORDS_PATH)[["Recordings", "age", "gender", "bmi", "lying_blood_pressure_systolic"]],
                   train_with_cross_val=True,
                   cv=KFold(n_splits=5, shuffle=True, random_state=1),
                   log=True,
                   display=True,
                   targets=["age"])

    print("Done")



def eval_one_epoch(model: torch.nn.Module,
                   data_loader: Iterable,
                   device: torch.device,
                   epoch: int,
                   log_writer=None,
                   args=None,
                   display_reconstruction=False,
                   cal_pearson=False):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")  # manages logging to print averaged metrics at once
    header = 'Epoch: [{}]'.format(epoch + 1)
    print_freq = 50

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Loop over batches
    losses = []
    recon_losses = []
    sex_losses = []
    age_losses = []
    ahi_losses = []
    sqi_losses = []
    corrs = []
    xcorrs = []
    step_end_time = time.time()
    for data_iter_step, (samples, _ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step_start_time = time.time()
        if isinstance(samples, list):
            samples = torch.stack(samples, dim=0)
        if samples is None:
            continue

        # move samples to device
        samples = samples.to(device, non_blocking=True)

        # retrieve features for supervision if needed
        if args.multi_loss:
            sex_target, age_target, ahi_target, sqi_target = get_supervised_targets_for_samples(_ids)
        else:
            age_target, sex_target, ahi_target, sqi_target = None, None, None, None

        # forward pass
        forward_start_time = time.time()
        with torch.no_grad():  # evaluate with the same task as used for training
            loss_a, outputs, masks, padding_mask, _, target, recon_loss, age_loss, sex_loss, ahi_loss, sqi_loss = model(
                samples, mask_ratio=args.mask_ratio,
                various_masking_strategies=args.various_masking_strategies,
                age_target=age_target, sex_target=sex_target,
                ahi_target=ahi_target, sqi_target=sqi_target)

        forward_end_time = time.time()
        loss_value = loss_a.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping evaluation".format(loss_value))
            metric_logger.update(loss=loss_value)
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # log step-level metrics
        losses.append(loss_value)
        recon_losses.append(recon_loss.item())
        sex_losses.append(sex_loss.item())
        age_losses.append(age_loss.item())
        ahi_losses.append(ahi_loss.item())
        sqi_losses.append(sqi_loss.item())
        metric_logger.update(loss=loss_value)

        # compute effective mask and loss channels for other metrics and display
        relative_loss_channels = [args.input_channels.index(ch) for ch in args.loss_channels]
        masks = masks * (1 - padding_mask)

        evaluation_start_time = time.time()
        if display_reconstruction and (data_iter_step + 1) % 10 == 0:
            function_name = f"display_reconstruction_{args.architecture}"
            # Get the function from the current module
            reconstruction_func = globals().get(function_name)
            reconstruction_func(samples, outputs, masks, args.patch_size, relative_loss_channels, model.unpatchify, "training_task_valset")

        if cal_pearson:
            # eval pearson correlation for batch:
            corr, xcorr, dtw = pcc_on_masked_patches(outputs, target, masks,
                       relative_loss_channels, model.unpatchify, model.patchify)  # arguments: x, target, masks
            metric_logger.update(corr=corr)
            metric_logger.update(xcorr=xcorr)
            corrs.append(corr.cpu().detach().numpy())
            xcorrs.append(xcorr.cpu().detach().numpy())
        evaluation_end_time = time.time()
        # end of batch

        # time tracking
        data_loading_time = step_start_time - step_end_time  # step_end_time refers to previous step
        step_end_time = time.time()
        step_total_time = step_end_time - step_start_time
        forward_time = forward_end_time - forward_start_time
        evaluation_time = evaluation_end_time - evaluation_start_time

        # Percentages
        percent_data_loading = 100 * data_loading_time / step_total_time
        percent_forward = 100 * forward_time / step_total_time
        percent_evaluation = 100 * evaluation_time / step_total_time

        # Log to wandb
        wandb.log({
            "eval_step_total_time": step_total_time,
            "eval_step_data_loading_time": data_loading_time,
            "eval_step_forward_time": forward_time,
            "eval_step_evaluation_time": evaluation_time,
            "eval_step_data_loading_%": percent_data_loading,
            "eval_step_forward_%": percent_forward,
            "eval_step_evaluation_%": percent_evaluation,
        })
        if 1: #data_iter_step >= MAX_STEPS_PER_EPOCH:
            break

    # end of epoch

    # compute epoch-level metrics
    wandb.log({"Val Total Loss per epoch": np.mean(losses),
               "Val Reconstruction Loss (MSE) per epoch": np.mean(recon_losses),
               "Val sex loss per epoch": np.mean(sex_losses),
               "Val age loss per epoch": np.mean(age_losses),
               "Val AHI loss per epoch": np.mean(ahi_losses),
               "Val SQI loss per epoch": np.mean(sqi_losses)
               })
    if corrs is not None:
        wandb.log({f"Val Pearson corr per epoch": np.nanmean(corrs),
                   "Val x-corr per epoch": np.nanmean(xcorrs) if xcorrs else None
                   })

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes() # for distributed train
    print("Validation averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None,
                    scheduler=None,  # default scheduler from Meta's framework
                    display_reconstruction=False,
                    cal_pearson=True):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")  # manages logging to print all metrics at once
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)
    print_freq = 50

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # set model epoch
    model.epoch = epoch
    losses = []
    recon_losses = []
    sex_losses = []
    age_losses = []
    ahi_losses = []
    sqi_losses = []
    latent_data = []
    step_end_time = time.time()
    for data_iter_step, (samples, _ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step_start_time = time.time()
        if isinstance(samples, list):
            samples = torch.stack(samples, dim=0)
        if samples is None:
            continue

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            if scheduler is None: # Meta's default
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # For debug
        if DEBUG:
            # load saved model and batch for debugging from os.path.join(args.output_dir, f"samples_{wandb.run.id}.pkl")
            with open(os.path.join(args.output_dir, f"samples_{wandb.run.id}.pkl"), 'rb') as f:
                samples = torch.load(f)
            with open(os.path.join(args.output_dir, f"masks_{wandb.run.id}.pkl"), 'rb') as f:
                masks = torch.load(f)
            samples = samples.to(device, non_blocking=True)
            misc.load_model(args=args, model=model,  model_without_ddp=model,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            model.train(True)

        # retrieve features for supervision if needed
        if args.multi_loss:
            sex_target, age_target, ahi_target, sqi_target = get_supervised_targets_for_samples(_ids)
        else:
            age_target, sex_target, ahi_target, sqi_target = None, None, None, None

        # increase masking ratio if half of the epochs are done:
        if args.increase_mask_ratio and epoch > args.epochs / 2:
            mask_ratio = max(0.9, args.mask_ratio + 0.1)
        else:
            mask_ratio = args.mask_ratio

        # forward pass (optionally with AMP for L40S)
        samples = samples.to(device, non_blocking=True)
        training_start = time.time()
        with torch.cuda.amp.autocast(enabled=(loss_scaler is not None)):
            loss, outputs, masks, padding_mask, latent, target, recon_loss, age_loss, sex_loss, ahi_loss, sqi_loss = model(
                samples, mask_ratio=mask_ratio,
                # various_masking_strategies=args.various_masking_strategies,
                age_target=age_target, sex_target=sex_target,
                ahi_target=ahi_target, sqi_target=sqi_target)

        loss_value = loss.item()
        # print(f"Training step {data_iter_step}, loss: {loss_value:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # save checkpoint and data
            misc.save_model(args=args, model=model,  model_without_ddp=model,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            # save samples and masks
            with open(os.path.join(args.output_dir, f"samples_{wandb.run.id}.pkl"), 'wb') as f:
                torch.save(samples, f)
            with open(os.path.join(args.output_dir, f"masks_{wandb.run.id}.pkl"), 'wb') as f:
                torch.save(masks, f)
            sys.exit(1)

        # log step-level metrics (every N steps to reduce GPU sync from .item() and wandb)
        log_this_step = (data_iter_step % getattr(args, 'log_every_n_steps', LOG_EVERY_N_STEPS)) == 0
        losses.append(loss_value)
        recon_losses.append(recon_loss.item())
        sex_losses.append(sex_loss.item())
        age_losses.append(age_loss.item())
        ahi_losses.append(ahi_loss.item())
        sqi_losses.append(sqi_loss.item())
        if log_this_step:
            wandb.log({"Total loss per batch": loss_value,
                       "Reconstruction loss (MSE) per batch": recon_losses[-1],
                       "Sex loss per batch": sex_losses[-1],
                       "Age loss per batch": age_losses[-1],
                       "AHI loss per batch": ahi_losses[-1],
                       "SQI loss per batch": sqi_losses[-1]})

        loss = loss / accum_iter
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in gradient of {name}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if (data_iter_step + 1) % accum_iter == 0:
                if scheduler == 'ReduceLROnPlateau_MinTrainLoss':
                    scheduler.step(loss)
        training_end = time.time()

        # torch.cuda.synchronize()  # check if speed up when commented out
        metric_logger.update(loss=loss_value)

        # perform further evaluations on ability to reconstruct masked train data (optional)
        relative_loss_channels = [args.input_channels.index(ch) for ch in args.loss_channels]
        if display_reconstruction:  # and (data_iter_step + 1) % 10 == 0:
            function_name = f"display_reconstruction_{args.architecture}"
            # Get the function from the current module
            reconstruction_func = globals().get(function_name)
            # take the masked patches only from non-padded areas
            masks = masks * (1 - padding_mask)
            reconstruction_func(samples, outputs, masks,  args.patch_size, relative_loss_channels, model.unpatchify, "training_task_trainset")

        # Concatenate embeddings only when requested (get_latent_dataframe does .cpu() and syncs GPU)
        collect_latent_n = getattr(args, 'collect_latent_every_n_steps', COLLECT_LATENT_EVERY_N_STEPS)
        if collect_latent_n > 0 and (data_iter_step % collect_latent_n) == 0:
            latent_data.append(get_latent_dataframe(_ids,
                                                    latent,
                                                    use_cls_token=run_config["use_cls_token"],
                                                    pooling_method=run_config["pooling_method"]))

        # log current learning rate (every N steps to reduce sync)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if log_this_step:
            wandb.log({f"learning rate per step": lr})

        loss_value_reduce = misc.all_reduce_mean(loss_value)  # for distributed train
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        # time tracking
        data_loading_time = step_start_time - step_end_time  # step_end_time refers to previous step
        step_end_time = time.time()
        step_total_time = step_end_time - step_start_time
        training_time = training_end - training_start

        # Percentages
        percent_data_loading = 100 * data_loading_time / step_total_time
        percent_training = 100 * training_time / step_total_time

        # Log to wandb (every N steps to reduce overhead)
        if log_this_step:
            wandb.log({
                "train_step_total_time": step_total_time,
                "train_step_data_loading_time": data_loading_time,
                "train_step_training_time": training_time,
                "train_step_data_loading_%": percent_data_loading,
                "train_step_training_%": percent_training,
            })

        if OVERFIT_EXP:
            break   # train on 1 single input to see overfitting
        # end of batch

        if MAX_STEPS_PER_EPOCH is not None and data_iter_step >= MAX_STEPS_PER_EPOCH:
            break

    # end of epoch

    # compute epoch-level metrics
    wandb.log({"Train Total Loss per epoch": np.mean(losses),
               "Train Reconstruction Loss (MSE) per epoch": np.mean(recon_losses),
               "Train sex loss per epoch": np.mean(sex_losses),
               "Train age loss per epoch": np.mean(age_losses),
               "Train AHI loss per epoch": np.mean(ahi_losses),
               "Train SQI loss per epoch": np.mean(sqi_losses)
               })

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes() # for distributed train
    print("Train averaged stats:", metric_logger)

    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    ######################################
    # program init
    ######################################
    # set run config
    run_config = {"architecture": ARCHITECTURE,
                  "model": MODEL,
                  "data_path": PATH_FOR_PROCESSED_DATA,
                  "dataset": 'HPP sleep raw data',
                  "patch_size": PATCH_SIZE,
                  "mask_ratio": MASK_PROB,
                  "increase_mask_ratio": INCREASE_MASK_RATIO,
                  "various_masking_strategies": VARIOUS_MASKING_STRATEGIES,
                  "augmentations": AUGMENTATIONS,
                  "input_size": INPUT_SIZE,
                  "n_items": N_ITEMS,
                  "zscore": ZSCORE,
                  "emb_dim": EMBEDED_DIM,
                  "base_learning_rate": LR,
                  "patience": PATIENCE,
                  "weight_decay": WD,
                  "warmup_epochs": WARMUP_EPOCHS,
                  "epochs": EPOCHS,
                  "batch_size": BATCH_SIZE,
                  "norm_pix_loss": NORM_PIX_LOSS,
                  "Pytorch": torch.__version__,
                  "input_channels": INPUT_CHANNELS,
                  "n_channels": len(INPUT_CHANNELS),
                  "multi_loss": MULTI_LOSS,
                  "loss_channels": LOSS_CHANNELS,
                  "loss_version": LOSS_VERSION,
                  "eval_model": EVAL_MODEL,
                  "use_cls_token": USE_CLS_TOKEN,
                  "pooling_method": POOLING_METHOD,
                  "display_reconstruction": True,
                  "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                  "n_GPUs": torch.cuda.device_count(),
                  "scheduler": SCHEDULER,
                  "num_workers": NUM_WORKERS,
                  "prefetch_factor": PREFETCH_FACTOR,
                  "use_amp": USE_AMP,
                  "log_every_n_steps": LOG_EVERY_N_STEPS,
                  "collect_latent_every_n_steps": COLLECT_LATENT_EVERY_N_STEPS,
                  "output_dir": os.path.join(PERSONAL_DIR, 'output_dir')}

    # get args
    args = get_args_parser(run_config)  # initialize the parser and define default arguments from run_config
    if OVERFIT_EXP:
        args.set_defaults(batch_size=1, accum_iter=1, num_workers=0, eval_model=True, distributed=False)
    args = args.parse_args()  # check if arguments were received in command line, and if yes use them

    # in case of sweep a combination of input channels is sent instead of the channel indexes
    if args.n_channels != len(run_config["input_channels"]):
        # loaded_combinations = load_combinations(f'channels_combinations/combinations_on_ch{args.loss_channels[0]}.json')
        # comb_idx = args.config['input_channels']
        # args.input_channels = loaded_combinations[comb_idx]
        if args.n_channels == 1:
            args.input_channels = [2]
        elif args.n_channels == 3:
            args.input_channels = [0, 1, 2]
        print(f"Input channels set to {args.input_channels} based on n_channels={args.n_channels}")

    # Initialize WandB
    if EPOCHS > MIN_EPOCHS_FOR_RESUME:  # run with checkpoints for training that are longer than 12 hours
        if MAX_STEPS_PER_EPOCH is not None and MAX_STEPS_PER_EPOCH <= 100 or N_ITEMS is not None:
            wandb.init(project=f"train_SSL_for_sleep", config=run_config.copy(), dir=PERSONAL_DIR, resume="allow",
                id=f"new2_SSLsubset_{args.architecture}_in{len(args.input_channels)}_out_{SIGNALS[LOSS_CHANNELS[0]]}_p{args.patch_size}_m{args.mask_ratio}_lr{args.base_learning_rate}_sc{args.scheduler}_loss_{args.loss_version}_noScaler")  # CHECK IF TO USE misc.load_model() INSTEAD

        else:
            wandb.init(project=f"train_SSL_for_sleep", config=run_config.copy(), dir=PERSONAL_DIR, resume="allow",
                # id=f"final_{args.num_workers}w{args.prefetch_factor}p{args.batch_size}b_SSL_{args.architecture}_in{len(args.input_channels)}_out_{SIGNALS[LOSS_CHANNELS[0]]}_p{args.patch_size}_m{args.mask_ratio}_lr{args.base_learning_rate}_sc{args.scheduler}_loss_{args.loss_version}_strat{args.various_masking_strategies}")  # CHECK IF TO USE misc.load_model() INSTEAD
                id = f"final1000steps_{args.emb_dim}D_{args.num_workers}w{args.prefetch_factor}p{args.batch_size}b_SSL_{args.architecture}_in{len(args.input_channels)}_out_{SIGNALS[LOSS_CHANNELS[0]]}_s{int(args.input_size / 125)}_p{args.patch_size}_m{args.mask_ratio}_lr{args.base_learning_rate}_sc{args.scheduler}_loss_{args.loss_version}_strat{args.various_masking_strategies}")  # CHECK IF TO USE misc.load_model() INSTEAD
                # id = f"final_{args.emb_dim}D_8w2p{args.batch_size}b_SSL_{args.architecture}_in{len(args.input_channels)}_out_{SIGNALS[LOSS_CHANNELS[0]]}_s{int(args.input_size / 125)}_p{args.patch_size}_m{args.mask_ratio}_lr{args.base_learning_rate}_sc{args.scheduler}_loss_{args.loss_version}_strat{args.various_masking_strategies}")  # CHECK IF TO USE misc.load_model() INSTEAD
                # id = f"final_8w2p{args.batch_size}b_SSL_{args.architecture}_in{len(args.input_channels)}_out_{SIGNALS[LOSS_CHANNELS[0]]}_p{args.patch_size}_m{args.mask_ratio}_lr{args.base_learning_rate}_sc{args.scheduler}_loss_{args.loss_version}_strat{args.various_masking_strategies}")  # CHECK IF TO USE misc.load_model() INSTEAD

    else:  # regular single run
        wandb.init(project=f"train_SSL_for_sleep", config=run_config.copy(), dir=PERSONAL_DIR, resume="allow")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # check if a checkpoint is saved for this run, if yes:
    # load the checkpoint and update args.resume with the checkpoint file name
    file_path = os.path.join(args.output_dir, f"checkpoint-{wandb.run.id}.pth")
    if os.path.isfile(file_path):
        args.resume = file_path

    main(args)

    # close wandb
    wandb.finish()