import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from model.utils import *


class SafeDataLoaderWrapper:
    """
    Wrapper that handles DataLoader worker errors gracefully and supports len().

    Args:
        data_loader: PyTorch DataLoader
        skip_on_error: If True, skip problematic batches; if False, re-raise
        verbose: If True, print warnings when skipping batches
    """

    def __init__(self, data_loader, skip_on_error=True, verbose=True):
        self.data_loader = data_loader
        self.skip_on_error = skip_on_error
        self.verbose = verbose

    def __iter__(self):
        """Generator that yields batches, skipping on error."""
        batch_idx = 0
        while True:
            try:
                for batch in self.data_loader:
                    yield batch
                    batch_idx += 1
                break
            except RuntimeError as e:
                if self.skip_on_error and ("DataLoader worker" in str(e) or "OOM" in str(e)):
                    if self.verbose:
                        print(f"Warning: DataLoader worker error at batch {batch_idx}, skipping...")
                    continue
                else:
                    raise
            except Exception as e:
                if self.skip_on_error and self.verbose:
                    print(f"Warning: Unexpected error in DataLoader at batch {batch_idx}, skipping: {e}")
                    continue
                else:
                    raise

    def __len__(self):
        """Return the length of the underlying data loader."""
        return len(self.data_loader)

def has_anomalies(sample, threshold=5):
    """Check if a sample has abnormal values using z-score"""
    if isinstance(sample, torch.Tensor):
        data = sample.cpu().numpy()
    else:
        data = sample

    # mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return False

    # z_scores = np.abs((data - mean) / std)
    # return np.any(z_scores > threshold)
    return np.any(np.abs(data) > threshold*10)


def safe_collate(batch, max_sample_mb=300):
    """
    A safe collate_fn that:
    - skips None or corrupted samples
    - skips samples larger than max_sample_mb
    - returns the batch as a list (no stacking)
    """
    cleaned_batch = []
    cleaned_ids = []

    for i, (sample, id) in enumerate(batch):
        # Skip None samples
        if sample is None:
            continue

        # Only handle tensors for simplicity
        if isinstance(sample, torch.Tensor):
            size_mb = sample.nelement() * sample.element_size() / 1e6
            # print(size_mb)
            if size_mb > max_sample_mb:
                print(f"[WARNING] Sample {i} is too large ({size_mb:.1f} MB). Skipping.")
                continue

        cleaned_batch.append(sample)
        cleaned_ids.append(id)

    return cleaned_batch, cleaned_ids


def collate_fn_skip_anomalies(batch, threshold=10):
    """
    Remove samples with anomalies from the batch before stacking.
    Returns variable batch size if anomalies are found.
    """
    # Filter out None values (failed samples)
    batch = [item for item in batch if item is not None]

    # Filter out samples with anomalies
    filtered_batch = [
        item for item in batch
        if not has_anomalies(item[0], threshold=threshold)
    ]

    # If all samples were filtered out, return original batch or handle gracefully
    if len(filtered_batch) == 0:
        print("Warning: All samples in batch were anomalous!")
        filtered_batch = batch  # fallback to original
        return None, None

    # Stack the data and targets
    data = torch.stack([item[0] for item in filtered_batch])
    ids = [item[1] for item in filtered_batch]

    return data, ids

def collate_fn_EffNet(batch):
    """
    Custom collate function to pad sequences to the same length within a batch.
    """
    inputs, ids, labels = zip(*batch)
    # remove None elements from inputs
    inputs = [x.T for x in inputs if x is not None]
    ids = [id for id in ids if id is not None]
    labels = [label for label in labels if label is not np.nan]
    # make all inputs with same 2nd dimensions (time) by padding with zeros
    padded_inputs = pad_sequence(inputs, batch_first=False, padding_value=0)
    # reshape to (BATCH, CHANNELS, seq_len)
    padded_inputs = padded_inputs.permute(1, 2, 0)
    return padded_inputs, ids, labels


def collate_fn_MAE1d(batch):
    """
    Custom collate function to pad sequences to the same length within a batch.
    """
    inputs, ids = zip(*batch)
    # remove None elements from inputs
    inputs = [x.T for x in inputs if x is not None]
    ids = [id for id in ids if id is not None]
    # make all inputs with same 2nd dimensions (time) by padding with zeros
    padded_inputs = pad_sequence(inputs, batch_first=False, padding_value=0)
    # reshape to (BATCH, CHANNELS, seq_len)
    padded_inputs = padded_inputs.permute(1, 2, 0)
    return padded_inputs, ids


def collate_fn_MAEspec(batch): 
    """
    Custom collate function to pad sequences to the same length within a batch.
    """
    inputs, ids = zip(*batch) #(B, ch, height, width)
    # remove None elements from inputs
    inputs = [x.T for x in inputs if x is not None] #(B, width, height, ch)
    ids = [id for id in ids if id is not None]
    # make all inputs with same 2nd dimensions (time) by padding with zeros
    padded_inputs = pad_sequence(inputs, batch_first=False, padding_value=0)
    # reshape to (BATCH, CHANNELS, (h,w)))
    padded_inputs = padded_inputs.permute(1, 3, 2, 0)
    return padded_inputs, ids


class RandomTimeMask:
    """ Randomly masks a portion of the time series. """
    def __init__(self, mask_ratio=0.1):
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        B, C, T = x.shape  # (Batch, Channels, Time)
        num_mask = int(T * self.mask_ratio)
        for i in range(B):
            mask_start = random.randint(0, T - num_mask)
            x[i, :, mask_start:mask_start + num_mask] = 0  # Zero out a portion
        return x

class Jitter:
    """ Adds small Gaussian noise to the signal. """
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class RandomScaling:
    """ Scales the amplitude of the signal randomly. """
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, x):
        scale = random.uniform(*self.scale_range)
        return x * scale

class TimeWarp:
    """ Applies slight stretching or compressing in time. """
    def __init__(self, warp_strength=0.1):
        self.warp_strength = warp_strength

    def __call__(self, x):
        B, C, T = x.shape
        new_T = int(T * (1 + random.uniform(-self.warp_strength, self.warp_strength)))
        x_resampled = torch.nn.functional.interpolate(x, size=new_T, mode='linear', align_corners=False)
        return x_resampled[:, :, :T] if new_T > T else torch.nn.functional.pad(x_resampled, (0, T - new_T))

class RandomChannelDropout:
    """ Randomly drops an entire channel with a probability. """
    def __init__(self, drop_prob=0.2):
        self.drop_prob = drop_prob

    def __call__(self, x):
        B, C, T = x.shape
        for i in range(B):
            if random.random() < self.drop_prob:
                channel_idx = random.randint(0, C - 1)
                x[i, channel_idx, :] = 0  # Zero out a channel
        return x


class TimeSeriesTransform:
    """ Custom Transform for Normalizing Multi-Channel Time Series Data.
     Defaults to mean and S.D. computed on 1000 samples from the training set (10Hz dataset)."""
    def __init__(self,
                 mean=torch.tensor([[8.6755e+01], [5.5093e+01], [2.3232e+03], [1.7623e+03], [1.8673e+01], [1.8239e+00], [3.5644e+01]]),
                 std=torch.tensor([[2.7840e+01], [1.9883e+01], [8.0125e+02], [6.3571e+02], [1.2794e+03], [1.2164e+00], [1.1477e+01]]),
                 channels=[0, 1, 2, 3, 4, 5, 6]):
        #
        self.mean = mean[channels]
        self.std = std[channels]

    def __call__(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return x.sub_(self.mean).div_(self.std)  #(x - self.mean) / self.std


class TemporalChannelwiseZScore:
    """
    Transform for Normalizing Multi-Channel Time Series Data using Temporal Channel-wise Z-scoring.
    Computes mean and standard deviation per channel over the time dimension for each sample.
    """
    def __init__(self, channels=None):
        self.channels = channels  # Select specific channels if provided

    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean across time
        std = x.std(dim=-1, unbiased=False, keepdim=True)  # Compute std across time
        std = std.clamp(min=1e-8)  # Avoid division by zero
        return (x - mean) / std

class RawSleepDataset(Dataset):
    def __init__(self, directory, architecture, split_dir, channels=[0,1,2,3,4,5,6], split="gold_train",
                 transform=None, compute_stats=False, n_items=None, segment_length=None):
        """
        For recording tensors saved in shape of [C, seq_length], as done for MAE preprocessing
        get_item() returns one random night of monitoring at each iteration
        Args:
            directory (string): Directory with all the sleep rawdata
            architecture (string): 'MAE' or other to define which type of preprocessing to apply.
            split_dir (string): e.g. 'randomstate1' to define which directory to extract the split from.
            channels (list): list of integers to define which channels to use.
            split (string): 'gold_all', 'gold_train', 'gold_val', 'train', 'val' or 'test' to define which part of the data to retrieve.
            transform (torchvision.transforms.Compose): Transformations for the dataset.
            compute_stats (bool): If True, compute mean & std (used for training).
        """
        self.architecture = architecture
        self.directory = directory
        self.channels = channels
        self.transform = transform
        self.length = segment_length

        # Load IDs for the desired split
        if split == 'gold_all':
            train_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_train_ids_{split_dir}.csv')
            )["RegistrationCode"].values
            val_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_val_ids_{split_dir}.csv')
            )["RegistrationCode"].values
            # take the union of all
            self.subset_ids = np.union1d(train_ids, val_ids)
        else:
            self.subset_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'{split}_ids_{split_dir}.csv')
            )["RegistrationCode"].values
        self.subset_ids = [f.split('_')[1] for f in self.subset_ids]
        self.file_names = [f for f in os.listdir(self.directory) if f.startswith(("preprocessed_"))]
        self.ids = self._extract_ids(self.file_names)
        self.recordings = self._extract_recordings(self.file_names)

        # Compute mean and std for training split
        if compute_stats and "train" in split:
            self.mean, self.std = self._compute_stats()  # each shape: (Channels, 1)
        else:
            self.mean, self.std = None, None  # Test/Val should use predefined mean/std

    def _compute_stats(self):
        """
        Computes mean and standard deviation across all recordings in the dataset.
        Only called for the training split.
        """
        count = 0
        mean = None
        M2 = None  # Sum of squares of differences from the current mean

        for i, recording in tqdm(enumerate(self.recordings)):
            sample_filename = os.path.join(self.directory, f'preprocessed_{recording}')
            tensor_data = torch.load(sample_filename)  # Shape: (num_channels, seq_length) or [n_seg, channels, segment_length] or [segment_legth,]
            # convert all to be with shape:  (num_channels, seq_length)
            if tensor_data.ndim == 1:  # PAT segments for example
                tensor_data = tensor_data.unsqueeze(0)
            elif tensor_data.ndim == 3:  # segmented night [n_items, c, segment_length]
                # concatenate all segments to get a 2D tensor of size [C, n_seg x segment_length]
                tensor_data = tensor_data.permute(1, 0, 2).reshape(tensor_data.shape[1], -1)

            # Process in smaller batches to keep memory low
            for segment in torch.split(tensor_data, 10000, dim=1):  # split by time chunks
                seg_count = segment.shape[1]
                seg_mean = segment.mean(dim=1, keepdim=True)
                seg_var = segment.var(dim=1, unbiased=False, keepdim=True)

            if mean is None:
                mean = seg_mean
                M2 = seg_var * seg_count
                count = seg_count
            else:
                delta = seg_mean - mean
                total_count = count + seg_count

                mean += delta * seg_count / total_count
                M2 += seg_var * seg_count + (delta ** 2) * count * seg_count / total_count
                count = total_count

            if i == 1000:
                break

        std = torch.sqrt(M2 / count + 1e-8)  # Add epsilon for numerical stability
        return mean, std

    def _extract_ids(self, file_names):
        """
        Extracts and returns unique IDs from file names.
        """
        ids = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]  # Assumes format of "preprocessed_ID__visitdate__night"
            # check if the ID is included in the desired split
            if person_id in self.subset_ids and person_id.isdigit():
                ids.append(int(person_id))
        unique_ids = list(set(ids))
        return unique_ids
    

    def _extract_recordings(self, file_names):
        """
        Extracts and returns unique recordings (in format ID_visitdate_night) from file names.
        """
        ids_and_dates = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]
            visit_date = file_name.split("__")[1]
            if "1night" not in self.directory:
                night_idx = file_name.split("__")[2]
                if person_id in self.subset_ids:
                    ids_and_dates.append(f"{person_id}__{visit_date}__{night_idx}")
            else:
                if person_id in self.subset_ids:
                    ids_and_dates.append(f"{person_id}__{visit_date}")
        unique_recordings = list(set(ids_and_dates))
        return unique_recordings

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        try:
            if idx >= len(self.recordings):
                raise IndexError(f"Index {idx} out of range")
            # get the tensor saved in the preprocessed file
            sample_filename = os.path.join(self.directory, f'preprocessed_{self.recordings[idx]}')
            tensor_data = torch.load(sample_filename)
            if np.isnan(tensor_data).any() or np.isinf(tensor_data).any():
                data = np.nan_to_num(tensor_data)  # Replace NaN with 0
            if tensor_data.ndim == 1:  # PAT segments for example
                tensor_data = tensor_data.unsqueeze(0)
            elif tensor_data.ndim == 3:  # segmented night [n_items, c, segment_length]
                # concatenate all segments to get a 2D tensor of size [C, n_items x segment_length]
                tensor_data = tensor_data.permute(1, 0, 2).reshape(tensor_data.shape[1], -1)

            if tensor_data.shape[0] != len(self.channels):
                tensor_data = tensor_data[self.channels, :]

            # Apply transforms (e.g., standard normalization)
            if self.transform:
                tensor_data = tensor_data.to(DEVICE)  # move to GPU is available
                tensor_data = self.transform(tensor_data)
                tensor_data = tensor_data.to('cpu')  # move back to CPU for saving

            # Crop signal to be of INPUT_SIZE
            if tensor_data.shape[1] > self.length:
                tensor_data = tensor_data[:, :self.length]
            # Pad the signal to be of INPUT_SIZE
            if tensor_data.shape[1] < self.length:
                tensor_data = torch.nn.functional.pad(tensor_data, (0, self.length - tensor_data.shape[1]))

            if 0:
                # Plot the signal
                from matplotlib import pyplot as plt
                signal = tensor_data[0, :].numpy()  # Convert to NumPy for plotting
                plt.figure(figsize=(10, 4))
                plt.plot(signal, label="PAT infra")
                plt.xlim([50000, 51250])
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.grid()
                plt.show()
            return tensor_data, self.recordings[idx]
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None  # Return None to signal skip


class RawSleepSubset(Dataset):
    def __init__(self, directory, architecture, split_dir, channels=[0,1,2,3,4,5,6], split="gold_train",
                 transform=None, compute_stats=False, n_items=100, segment_length=None):
        """
        Modified RawSleepDataset:
        - Loads a random subset of 100 files.
        - Caches each loaded file in RAM to avoid redundant loading.

        Args:
            directory (string): Directory with all the sleep rawdata.
            architecture (string): 'MAE' or other to define which type of preprocessing to apply.
            split_dir (string): Directory for the data split (e.g., 'randomstate1').
            channels (list): List of integers to define which channels to use.
            split (string): 'train', 'val' or 'test' to define the split.
            transform (torchvision.transforms.Compose): Transformations for the dataset.
            compute_stats (bool): If True, compute mean & std (used for training).
            n_files (int): Number of random files to include in the subset.
            seed (int): Random seed for reproducibility.
        """
        self.architecture = architecture
        self.directory = directory
        self.channels = channels
        self.transform = transform
        self.length = segment_length
        self.cache = {}  # Cache to store loaded tensors in memory

        # Load IDs for the desired split
        if split == 'gold_all':
            train_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_train_ids_{split_dir}.csv')
            )["RegistrationCode"].values
            val_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_val_ids_{split_dir}.csv')
            )["RegistrationCode"].values
            # take the union of all
            self.subset_ids = np.union1d(train_ids, val_ids)
        else:
            self.subset_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'{split}_ids_{split_dir}.csv')
            )["RegistrationCode"].values
        self.subset_ids = [f.split('_')[1] for f in self.subset_ids]

        # Get list of all files matching the subset IDs
        self.file_names = [f for f in os.listdir(self.directory) if f.startswith("preprocessed_")]
        self.file_names = [f for f in self.file_names if f.split("_")[1] in self.subset_ids]

        # Randomly select a subset of n_files
        if len(self.file_names) > n_items:
            np.random.seed(SEED)  # Set random seed for reproducibility
            self.file_names = list(np.random.choice(self.file_names, n_items, replace=False))
            print(f"Selected a subset of {n_items} random files.")

        # Extract recording names for indexing
        self.recordings = self._extract_recordings(self.file_names)

        # Compute mean and std for training split (optional)
        if compute_stats and "train" in split:
            self.mean, self.std = self._compute_stats()
        else:
            self.mean, self.std = None, None

    def _extract_recordings(self, file_names):
        """
        Extracts and returns unique recordings (in format ID__visitdate__night) from file names.
        """
        ids_and_dates = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]
            visit_date = file_name.split("__")[1]
            if "1night" not in self.directory:
                night_idx = file_name.split("__")[2]
                ids_and_dates.append(f"{person_id}__{visit_date}__{night_idx}")
            else:
                ids_and_dates.append(f"{person_id}__{visit_date}")
        return list(set(ids_and_dates))

    def _compute_stats(self):
        """
        Computes mean and standard deviation across all recordings in the dataset.
        Only called for the training split.
        """
        count = 0
        mean = None
        M2 = None

        for recording in tqdm(self.recordings, desc="Computing stats"):
            tensor_data = self._load_tensor(recording)
            for segment in torch.split(tensor_data, 10000, dim=1):
                seg_count = segment.shape[1]
                seg_mean = segment.mean(dim=1, keepdim=True)
                seg_var = segment.var(dim=1, unbiased=False, keepdim=True)

                if mean is None:
                    mean = seg_mean
                    M2 = seg_var * seg_count
                    count = seg_count
                else:
                    delta = seg_mean - mean
                    total_count = count + seg_count
                    mean += delta * seg_count / total_count
                    M2 += seg_var * seg_count + (delta ** 2) * count * seg_count / total_count
                    count = total_count

        std = torch.sqrt(M2 / count + 1e-8)
        return mean, std

    def _load_tensor(self, recording):
        """
        Loads the tensor from file or cache it in memory.
        """
        if recording not in self.cache:
            file_path = os.path.join(self.directory, f'preprocessed_{recording}')
            tensor_data = torch.load(file_path, map_location="cpu")

            # Convert tensor to 2D (C, seq_length) format
            if tensor_data.ndim == 1:  # PAT segments
                tensor_data = tensor_data.unsqueeze(0)
            elif tensor_data.ndim == 3:  # segmented night [n_items, C, segment_length]
                tensor_data = tensor_data.permute(1, 0, 2).reshape(tensor_data.shape[1], -1)

            # Select specified channels
            if tensor_data.shape[0] != len(self.channels):
                tensor_data = tensor_data[self.channels, :]

            # Apply transforms
            if self.transform:
                tensor_data = self.transform(tensor_data)

            # Crop or pad to desired length
            if self.length:
                if tensor_data.shape[1] > self.length:
                    tensor_data = tensor_data[:, :self.length]
                else:
                    tensor_data = torch.nn.functional.pad(tensor_data, (0, self.length - tensor_data.shape[1]))

            # cache the loaded tensor after transformations
            self.cache[recording] = tensor_data
        return self.cache[recording]

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        try:
            recording = self.recordings[idx]
            return self._load_tensor(recording), recording
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None  # Return None to signal skip


class RawSleepDataset2(Dataset):
    def __init__(self, directory, architecture, split_dir, channels=[0, 1, 2, 3, 4, 5, 6], split="gold_train",
                 transform=None, n_items=None, segment_length=None, n_per_recording=None):
        """
        For recording tensors saved in shape of [n_items, channel, segment_length], as done for PaPaGei preprocessing
        get_item() returns one random segment at each iteration
        Args:
            directory (string): Directory with all the sleep rawdata
            architecture (string): 'MAE' or other to define which type of preprocessing to apply.
            split_dir (string): e.g. 'randomstate1' to define which directory to extract the split from.
            channels (list): list of integers to define which channels to use.
            split (string): 'train', 'val' or 'test' to define which part of the data to retrieve.
            transform (torchvision.transforms.Compose): Transformations for the dataset.
            compute_stats (bool): If True, compute mean & std (used for training).
        """
        self.architecture = architecture
        self.directory = directory
        self.channels = channels
        self.transform = transform
        self.n_per_recording = n_per_recording
        self.sample_indices = [71, 3, 138, 102, 82, 56, 31, 42, 7, 80, 2, 53, 78, 38, 119, 44, 63, 83, 48, 85, 22, 94,
                               65, 86, 55, 20, 118, 73, 124, 133, 4, 9, 135, 24, 50, 37, 96, 36, 123, 106, 99, 10, 134,
                               126, 92, 43, 120, 62, 87, 26, 39, 97, 13, 52, 66, 12, 27, 101, 47, 113, 84, 29, 77, 130,
                               16, 67, 122, 125, 45, 108, 98, 61, 90, 100, 136, 104, 110, 68, 32, 11, 131, 107, 25, 19,
                               64, 51, 115, 18, 70, 0, 40, 129, 103, 91, 81, 112, 1, 76, 93, 75]


        # Load or create the index map (prefer compact .npz to avoid ~8GB .npy)
        compact_path = os.path.join(self.directory, f'index_map_{split}_compact.npz')
        legacy_path = os.path.join(self.directory, f'index_map_{split}.npy')
        if os.path.exists(compact_path):
            data = np.load(compact_path, allow_pickle=True)
            self.file_paths = data['paths'].tolist()  # unique paths
            self._indices = data['indices']          # int index into file_paths per sample
            self.file_names = self.file_paths
        elif os.path.exists(legacy_path):
            # Load legacy 8GB-style .npy and convert to compact (then optionally save for next time)
            self.index_map = np.load(legacy_path, allow_pickle=True)
            unique_paths = list(dict.fromkeys([item[0] for item in self.index_map]))
            path_to_idx = {p: i for i, p in enumerate(unique_paths)}
            self._indices = np.array([path_to_idx[item[0]] for item in self.index_map], dtype=np.uint32)
            self.file_paths = unique_paths
            self.file_names = self.file_paths
            np.savez_compressed(compact_path, paths=np.array(unique_paths, dtype=object), indices=self._indices)
            print('Saved compact index_map to:', compact_path)
            del self.index_map  # free the large array
        else:
            # Build index from disk
            if split == 'gold_all':
                train_ids = pd.read_csv(
                    os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_train_ids_{split_dir}.csv')
                )["RegistrationCode"].str.split('_').str[1].values
                val_ids = pd.read_csv(
                    os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_val_ids_{split_dir}.csv')
                )["RegistrationCode"].str.split('_').str[1].values
                subset_ids = np.union1d(train_ids, val_ids)
            else:
                split_path = os.path.join(PARENT_DIR, 'data_splits', split_dir, f'{split}_ids_{split_dir}.csv')
                subset_ids = pd.read_csv(split_path)["RegistrationCode"].str.split('_').str[1].values
            rec_list = os.listdir(self.directory)
            rec_list = [f for f in rec_list if f.startswith("preprocessed_") and f.split("_")[1] in subset_ids]
            rec_list.sort()

            all_paths = []
            self.file_names = []
            self.recordings = []
            for rec_name in tqdm(rec_list, desc=f"Indexing {split} files"):
                file_list = os.listdir(os.path.join(self.directory, rec_name))
                file_list = [os.path.join(self.directory, rec_name, f) for f in file_list if f.endswith(".pt")]
                self.file_names.append(file_list)
                self.recordings.append(rec_name)
                all_paths.extend(file_list)
            unique_paths = list(dict.fromkeys(all_paths))
            path_to_idx = {p: i for i, p in enumerate(unique_paths)}
            self._indices = np.array([path_to_idx[p] for p in all_paths], dtype=np.uint32)
            self.file_paths = unique_paths
            np.savez_compressed(compact_path, paths=np.array(unique_paths, dtype=object), indices=self._indices)
            print('Saved compact index_map to:', compact_path)

        if n_per_recording is not None:
            pass
            # # list only files that have at least max(self.sample_indices) segments, to check that use index_map
            # valid_files = set()
            # for item in self.index_map:
            #     if int(item[2]) >= max(self.sample_indices):
            #         valid_files.add(item[0])
            # self.file_names = [f for f in self.file_names if f in valid_files]

    def __len__(self):
        if self.n_per_recording is not None:
            return len(self.file_names) * self.n_per_recording
        else:
            return len(self._indices)

    def __getitem__(self, idx):
        try:
            if self.n_per_recording is not None:
                pass
                # # Map single index to recording file and segment number
                # recording_idx = idx // self.n_per_recording
                # segment_num = idx % self.n_per_recording
                # if recording_idx >= len(self.file_names):
                #     raise IndexError(f"Index {recording_idx} out of file range")
                # # Retrieve tensor data from file and random sample_idx
                # filename = self.file_names[recording_idx]
                # if self._file_name is None or filename != self._file_name:
                #     self._file_name = filename
                #     self._file = torch.load(os.path.join(self.directory, filename))
                # tensor_data = self._file[segment_num]
                # # Get n_per_recording random samples from the recording
                # # num_available = tensor_data.shape[0]
                # # sample_indices = random.sample(range(num_available), k=self.n_per_recording)
                # sample_indices = self.sample_indices[:self.n_per_recording]
                # sample_idx = sample_indices[segment_num]
                # if sample_idx >= len(tensor_data):
                #     raise IndexError(f"Index {sample_idx} out of segments range")
                # tensor_data = tensor_data[int(sample_idx)]  # Returns a single sample of shape [C, segment_length]

            else:
                if idx >= len(self._indices):
                    raise IndexError(f"Index {idx} out of range")
                file_idx = int(self._indices[idx])
                filename = self.file_paths[file_idx]
                recording = os.path.basename(os.path.dirname(filename))
                tensor_data = torch.load(filename)

            if np.isnan(tensor_data).any() or np.isinf(tensor_data).any():
                data = np.nan_to_num(tensor_data)  # Replace NaN with 0
            if tensor_data.ndim == 1:  # PAT segments for example
                tensor_data = tensor_data.unsqueeze(0)
            elif tensor_data.shape[0] != len(self.channels):
                tensor_data = tensor_data[self.channels, :]

            # Apply transforms (e.g., standard normalization)
            if self.transform:
                tensor_data = self.transform(tensor_data)

            if 0:
                # Plot the signal
                from matplotlib import pyplot as plt
                signal = tensor_data[0, :].numpy()  # Convert to NumPy for plotting
                plt.figure(figsize=(10, 4))
                plt.plot(signal, label="PAT infra")
                plt.xlim([0, 1250])
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.grid()
                plt.show()
            return tensor_data, recording
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None  # Return None to signal skip


class RawSleepSubset2(Dataset):
    def __init__(self, directory, architecture, split_dir, channels=[0, 1, 2, 3, 4, 5, 6], split="train",
                 transform=None, n_items=10000, segment_length=None):
        """
        Modified RawSleepDataset2:
        - Loads 10,000 random segments from the dataset.
        - Caches loaded tensors in RAM to avoid reloading.

        Args:
            directory (string): Directory with all the sleep rawdata.
            architecture (string): 'MAE' or other to define preprocessing type.
            split_dir (string): Directory for the data split (e.g., 'randomstate1').
            channels (list): List of integers defining which channels to use.
            split (string): 'train', 'val' or 'test' to define the split.
            transform (torchvision.transforms.Compose): Transformations for the dataset.
            n_items (int): Number of random segments to include in the subset.
        """
        self.architecture = architecture
        self.directory = directory
        self.channels = channels
        self.transform = transform
        self.cache = {}  # Cache to store loaded tensors in memory

        # Load or create the index map
        index_map_path = os.path.join(self.directory, f'index_map_{split}.npy')
        if os.path.exists(index_map_path):
            self.index_map = np.load(index_map_path, allow_pickle=True)
        else:
            self.index_map = self._create_index_map(split_dir, split)
            np.save(index_map_path, self.index_map)
            print('Saved index_map to file:', index_map_path)

        # Randomly select a subset of n_items
        if len(self.index_map) > n_items:
            np.random.seed(SEED)  # Set random seed for reproducibility
            selected_indices = np.random.choice(len(self.index_map), n_items, replace=False)
            self.index_map = self.index_map[selected_indices]
            print(f"Selected a subset of {n_items} random segments.")

    def _create_index_map(self, split_dir, split):
        """Create an index map by reading available data files."""
        # Load split subset
        if split == 'gold_all':
            train_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_train_ids_{split_dir}.csv')
            )["RegistrationCode"].str.split('_').str[1].values
            val_ids = pd.read_csv(
                os.path.join(PARENT_DIR, 'data_splits', split_dir, f'gold_val_ids_{split_dir}.csv')
            )["RegistrationCode"].str.split('_').str[1].values
            # take the union of all or train and val ids
            subset_ids = np.union1d(train_ids, val_ids)
        else:
            split_path = os.path.join(PARENT_DIR, 'data_splits', split_dir, f'{split}_ids_{split_dir}.csv')
            subset_ids = pd.read_csv(split_path)["RegistrationCode"].str.split('_').str[1].values
        file_list = os.listdir(self.directory)
        file_list = [f for f in file_list if f.startswith("preprocessed_") and f.split("_")[1] in subset_ids]

        file_names = []
        index_map = []
        for file_name in tqdm(file_list, desc=f"Indexing {split} files"):
            file_path = os.path.join(self.directory, file_name)
            try:
                tensor = torch.load(file_path, map_location="cpu")
                num_segments = tensor.shape[0]
                file_index = len(file_names)
                file_names.append(file_name)
                index_map.extend([(file_name, file_index, i) for i in range(num_segments)])
            except Exception as e:
                print(f"Skipping {file_name}: {e}")
        return np.array(index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        try:
            # Retrieve the filename and segment index from the index map
            filename, _, sample_idx = self.index_map[idx]

            # Load the tensor into RAM (only if not already cached)
            if filename not in self.cache:
                file_path = os.path.join(self.directory, filename)
                tensor_data = torch.load(file_path, map_location="cpu")
                # Select channels if needed
                if tensor_data.ndim == 2:  # i.e. (Num_segments, segment_length) For single-channel segments (e.g., PAT)
                    tensor_data = tensor_data.unsqueeze(1)
                elif tensor_data.shape[1] != len(self.channels):
                    tensor_data = tensor_data[:, self.channels, :]
                # Apply transforms (e.g., standard normalization)
                if self.transform:
                    tensor_data = self.transform(tensor_data)
                # Cache the loaded tensor
                self.cache[filename] = tensor_data

            return self.cache[filename][int(sample_idx)], filename.split("preprocessed_")[1]
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None  # Return None to signal skip

class SleepSpectogramDataset(Dataset):
    def __init__(self, directory, architecture, split_dir, input_size = (64, 4096), channels=[0, 1, 2, 3, 4, 5, 6], split="train"):
        """
        Args:
            directory (string): Directory with all the sleep rawdata
            architecture (string): 'MAE' or other to define which type of preprocessing to apply.
            split_dir (string): e.g. 'randomstate1' to define which directory to extract the split from.
            channels (list): list of integers to define which channels to use.
            split (string): 'train', 'val' or 'test' to define which part of the data to retrieve.
        """
        self.architecture = architecture
        self.directory = directory
        self.channels = channels
        self.height= input_size[0]
        self.width= input_size[1]
        self.subset_ids = pd.read_csv(
            os.path.join('data_splits', split_dir, f'{split}_ids_{split_dir}.csv')
        )["RegistrationCode"].values
        self.subset_ids = [f.split('_')[1] for f in self.subset_ids]
        self.file_names = [f for f in os.listdir(self.directory) if f.startswith(("preprocessed_"))]
        self.ids = self._extract_ids(self.file_names)
        self.recordings = self._extract_recordings(self.file_names)

    def _extract_ids(self, file_names):
        """
        Extracts and returns unique IDs from file names.
        """
        ids = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]  # Assumes format of "preprocessed_ID__visitdate__night"
            # check if the ID is included in the desired split
            if person_id in self.subset_ids and person_id.isdigit():
                ids.append(int(person_id))
        unique_ids = list(set(ids))
        return unique_ids

    def _extract_recordings(self, file_names):
        """
        Extracts and returns unique recordings (in format ID_visitdate_night) from file names.
        """
        ids_and_dates = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]
            visit_date = file_name.split("__")[1]
            if "1night" not in self.directory:
                night_idx = file_name.split("__")[2]
                if person_id in self.subset_ids:
                    ids_and_dates.append(f"{person_id}__{visit_date}__{night_idx}")
            else:
                if person_id in self.subset_ids:
                    ids_and_dates.append(f"{person_id}__{visit_date}")
        unique_recordings = list(set(ids_and_dates))
        return unique_recordings

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        # get the tensor saved in the preprocessed file
        sample_filename = os.path.join(self.directory, f'preprocessed_{self.recordings[idx]}')
        tensor_data = torch.load(sample_filename) # (ch,w,h)
        # Pad the third dimension to 4096 with zero at the end
        tensor_data = torch.nn.functional.pad(tensor_data, (0, 1)) 
        # extract only input channels, and crop to fit desired height (frequency) and width (time)
        tensor_data = tensor_data[self.channels, 0:min(self.height,tensor_data.shape[1]), 0:min(self.width, tensor_data.shape[-1])]
        return tensor_data, self.recordings[idx]
    

class RawSleepWithLabelsDataset(Dataset):
    def __init__(self, directory, labels_path, architecture, ids_path, label: str, channels=[0,1,2,3,4,5,6]):
        """
        Args:
            directory (string): Directory with all the sleep rawdata
            architecture (string): 'Supervised' or other to define which type of preprocessing to apply.
            split (string): 'train', 'val' or 'test' to define which part of the data to retrieve.
        """
        self.architecture = architecture
        self.directory = directory
        self.label = label
        self.channels = channels

        self._get_recordings(ids_path, labels_path)

    def _get_recordings(self, ids_path, labels_path):

        selected_ids = pd.read_csv(ids_path)["RegistrationCode"]

        recording_with_labels = pd.read_csv(
            labels_path
        )[["RegistrationCode", "Ids", "Recordings" ,self.label]]

        #discard recording with nan values in labels 
        recording_with_labels = recording_with_labels[~recording_with_labels[self.label].isnull()]
        
        self.recordings = recording_with_labels[recording_with_labels["RegistrationCode"].isin(selected_ids)]

        self.ids = self.recordings["Ids"].values
        self.labels = self.recordings[self.label].values
        self.recordings = self.recordings["Recordings"].values
        

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        # get the tensor saved in the preprocessed file
        sample_filename = os.path.join(self.directory, f'preprocessed_{self.recordings[idx]}')
        tensor_data = torch.load(sample_filename)
        tensor_data = tensor_data[self.channels, :]
        return tensor_data, self.recordings[idx], self.labels[idx]


class SleepSegmentsPerPersonDataset(Dataset):
    """
    Patient level positive set selection
    """
    def __init__(self, directory, architecture, split_dir, channels=[2], split="train", transform=None,
                 n_items=4, segment_length=30):
        """
        Args:
            directory (string): Directory with all the sleep rawdata
            architecture (string): 'MAE' or other to define which type of preprocessing to apply.
            split_dir (string): e.g. 'randomstate1' to define which directory to extract the split from.
            channels (list): list of integers to define which channels to use.
            split (string): 'train', 'val' or 'test' to define which part of the data to retrieve.
            transform (torchvision.transforms.Compose): Transformations for the dataset.
            n_items (int): Number of segments to split each recording into.
            segment_length (int): Length of each segment in number of samples (seconds*sampling_rate).
        """
        self.architecture = architecture
        self.directory = directory
        self.channels = channels
        self.transform = transform
        self.n_items = n_items
        self.segment_length = segment_length

        # Load IDs for the desired split
        self.subset_ids = pd.read_csv(
            os.path.join(PARENT_DIR, 'data_splits', split_dir, f'{split}_ids_{split_dir}.csv')
        )["RegistrationCode"].values
        self.subset_ids = [f.split('_')[1] for f in self.subset_ids]
        self.file_names = [f for f in os.listdir(self.directory) if f.startswith(("preprocessed_"))]
        self.ids = self._extract_ids(self.file_names)
        self.recordings = self._extract_recordings(self.file_names)

    def _extract_ids(self, file_names):
        """
        Extracts and returns unique IDs from file names.
        """
        ids = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]  # Assumes format of "preprocessed_ID__visitdate__night"
            # check if the ID is included in the desired split
            if person_id in self.subset_ids:
                ids.append(int(person_id))
        unique_ids = list(set(ids))
        return unique_ids

    def _extract_recordings(self, file_names):
        """
        Extracts and returns unique recordings (in format ID_visitdate_night) from file names.
        """
        ids_and_dates = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]
            visit_date = file_name.split("__")[1]
            if "1night" not in self.directory:
                night_idx = file_name.split("__")[2]
                if person_id in self.subset_ids:
                    ids_and_dates.append(f"{person_id}__{visit_date}__{night_idx}")
            else:
                if person_id in self.subset_ids:
                    ids_and_dates.append(f"{person_id}__{visit_date}")
        unique_recordings = list(set(ids_and_dates))
        return unique_recordings

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        # get the tensor saved in the preprocessed file
        sample_filename = os.path.join(self.directory, f'preprocessed_{self.recordings[idx]}')
        tensor_data = torch.load(sample_filename)
        tensor_data = tensor_data[self.channels, :]

        # Apply transforms (e.g., standard normalization)
        if self.transform:
            tensor_data = self.transform(tensor_data)

        # Split the tensor_data into segments of segment_length
        segments = np.array([tensor_data[:, i: (i + self.segment_length)]
                             for i in range(0, tensor_data.shape[1], self.segment_length)][:-1])

        # remove flat segments
        segments = [seg for seg in segments if torch.std(seg) > 1e-8]

        # keep only the first n_items, if there are less return None
        if len(segments) < self.n_items:
            return None, self.recordings[idx]
        else:
            return torch.stack(segments[:self.n_items]).squeeze(), self.recordings[idx]


class SleepSegmentDataset(Dataset):
    def __init__(self, directory, architecture, split_dir, channels=[2], split="train", transform=None, n_items=4, segment_length=30):
        """
        Generates segments from whole night recordings and returns them as a dataset.
        For each recording, it extracts segments of length segment_length and stores them in a list.
        Each segment is then returned with its corresponding recording ID.
        Args:
            directory (string): Directory with all the sleep rawdata.
            architecture (string): 'MAE' or other to define which type of preprocessing to apply.
            split_dir (string): Directory to extract the split from.
            channels (list): List of integers to define which channels to use.
            split (string): 'train', 'val' or 'test' to define which part of the data to retrieve.
            transform (torchvision.transforms.Compose): Transformations for the dataset.
            n_items (int): Number of segments to split each recording into.
            segment_length (int): Length of each segment in number of samples.
        """
        self.architecture = architecture
        self.directory = directory
        self.channels = channels
        self.transform = transform
        self.n_items = n_items
        self.segment_length = segment_length

        # Load IDs for the desired split
        self.subset_ids = pd.read_csv(
            os.path.join(PARENT_DIR, 'data_splits', split_dir, f'{split}_ids_{split_dir}.csv')
        )["RegistrationCode"].values
        self.subset_ids = [f.split('_')[1] for f in self.subset_ids]
        self.file_names = [f for f in os.listdir(self.directory) if f.startswith(("preprocessed_"))]
        self.recordings = self._extract_recordings(self.file_names)

        # Create list of (segment, ID) pairs
        self.segments = self._generate_segments()

    def _extract_recordings(self, file_names):
        """
        Extracts and returns unique recordings (in format ID_visitdate_night) from file names.
        """
        recordings = []
        for file_name in file_names:
            person_id = file_name.split("_")[1]
            visit_date = file_name.split("__")[1]
            if "1night" not in self.directory:
                night_idx = file_name.split("__")[2]
                if person_id in self.subset_ids:
                    recordings.append(f"{person_id}__{visit_date}__{night_idx}")
            else:
                if person_id in self.subset_ids:
                    recordings.append(f"{person_id}__{visit_date}")
        return list(set(recordings))  # Ensure uniqueness

    def _generate_segments(self):
        """
        Splits each recording into individual segments and stores them as (segment, ID) pairs.
        """
        all_segments = []
        for iter , recording_id in tqdm(enumerate(self.recordings)):
            sample_filename = os.path.join(self.directory, f'preprocessed_{recording_id}')
            tensor_data = torch.load(sample_filename)
            tensor_data = tensor_data[self.channels, :]

            # Apply transforms
            if self.transform:
                tensor_data = self.transform(tensor_data)

            # Split data into segments
            segments = [tensor_data[:, i: i + self.segment_length]
                        for i in range(0, tensor_data.shape[1], self.segment_length)][:-1]

            # Remove flat segments
            segments = [seg for seg in segments if torch.std(seg) > 1e-8]

            # Keep only the first n_items
            segments = segments[:self.n_items]

            # Store each segment with its corresponding ID
            all_segments.extend([(seg, recording_id) for seg in segments])

            # if iter == 10:
            #     break

        return all_segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Returns a single segment and its corresponding ID at each iteration.
        """
        segment, recording_id = self.segments[idx]
        return segment.squeeze().unsqueeze(0), recording_id  # Squeeze to remove unnecessary dimensions if needed



def MAEspec(**kwargs):
    train_dataset = SleepSpectogramDataset(**kwargs)
    return train_dataset

def MAE1d(**kwargs):  # dataset of entire night
    # check argument n_items in the kwargs and if not None call subset
    if kwargs.get('n_items') is not None:
        train_dataset = RawSleepSubset(**kwargs)
    else:
        train_dataset = RawSleepDataset(**kwargs)
    return train_dataset

def MAE1d_2(**kwargs): # dataset of few seconds segments
    # check argument n_items in the kwargs and if not None call subset
    if kwargs.get('n_items') is not None:
        train_dataset = RawSleepSubset2(**kwargs)
    else:
        train_dataset = RawSleepDataset2(**kwargs)
    return train_dataset

def Supervised1D(**kwargs):
    train_dataset = RawSleepWithLabelsDataset(**kwargs)
    return train_dataset

def PersonSegments1D(**kwargs):
    train_dataset = SleepSegmentsPerPersonDataset(**kwargs)
    return train_dataset

def Segments1D(**kwargs):
    train_dataset = SleepSegmentDataset(**kwargs)
    return train_dataset

# set recommended archs
MAE1d_dataset = MAE1d  # for adaptedMAE framework
MAE1d_papagei_dataset = MAE1d_2  # for adaptedMAE framework - if running on segments
MAE1d_segmented_dataset = Segments1D  # for adaptedMAE framework

MAE1d_on_night_dataset = MAE1d  # for adaptedMAE framework
MAE1d_on_segments_dataset = MAE1d_2  # for adaptedMAE framework - if running on segments
MAEspec_dataset = MAEspec  # for adaptedMAE framework
MAE_dataset = MAE1d  # for ssl_pretraining framework
Supervised1D_dataset = Supervised1D # for supervised framework (gili)
CNN_CL_dataset = PersonSegments1D  # for contrastive_learning framework
