import warnings
import sys
import os
import random
# import wandb
import pandas as pd
import numpy as np
# import logging
import matplotlib.pyplot as plt
# import gc
from scipy import signal
from tqdm import tqdm
import torch
# import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.utils.rnn import pad_sequence
# from timm.models.vision_transformer import Block

# from transformers import ViTMAEForPreTraining
from papagei.linearprobing.utils import resample_batch_signal
from papagei.segmentations import waveform_to_segments
# pyPPG / DotMap / preprocess_one_ppg_signal: imported inside compute_pyppg_biomarkers only,
# so the rest of this module (and `from timeseries_preprocessing import compute_pyppg_biomarkers`)
# does not require pyPPG at import time.

# global setup
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(PARENT_DIR)
PROJECT_DIR = '/net/mraid20/export/jafar/SleepFM/ssl_sleep/'
PERSONAL_DIR = '/net/mraid20/export/jafar/Sarah/ssl_sleep/'
PYPPG_OUTPUT_DIR = '/net/mraid20/export/jafar/SleepFM/pyPPG_output_gold/'
SEED = 30

# set dataset globals
RESAMPLE_RATE = 125  # Hz
SEC_IN_SEGMENT = [960*2, 960*4, 960*8]  #[10, 30, 60, 120]  # sec, length of each segment
MIN_DURATION = 3  # hrs, minimum duration of a recording to be considered as valid
WIN_LENGTH = 2**7  # ~10 sec window length for STFT (in original sampling rate)
OVERLAP = 0.5 # % overlap between windows for STFT
PATH_FOR_DATA = "/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep/timeseries/"
PATH_FOR_PROCESSED_DATA = PROJECT_DIR + f"preprocessed_data_SR_{RESAMPLE_RATE}Hz_gold"
SIGNALS = [
# # from Pulse oximetry sensor:
           'spo2',  # pulse oximetry sensor
           'heart_rate', # pulse oximetry sensor
# # from PAT sensor:
           # 'heart_rate_raw', #PAT sensor??
           # "raw_heart_rate_report"  # new in gold dataset
           'pat_infra',  # PAT sensor used by Itamar for CSR events and for HRV calculation
           # 'pat_amplitude',  # PAT sensor for resp. events detection
           # 'pat_lpf',
           # 'pat_view',
# # from wrist actigraphy:
           'actigraph', # (for sleep stages)
           # 'sleep_stage',
# # from chest RESBP sensor:
           'respiratory_movement',  # chest actigraphy with mean removal
           'body_position',  # from local means of chest actigraphy
           'snore_db'  # chest microphone
# # other files only in gold dataset:
            # 'channels'
            # "raw_events"
           ]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdirifnotexists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


class PreprocessSleepDataset(Dataset):
    def __init__(self, directory,
                 output_format='timeseries',
                 config=dict(target_sr=RESAMPLE_RATE,
                             min_duration=MIN_DURATION,
                             win_length=WIN_LENGTH,
                             overlap=OVERLAP)):
        """
        Args:
            directory (string): Directory with all the sleep rawdata
            output_format (string): 'timeseries' or 'images'
            config (dict): dictionary with the following keys
                            target_sr (int): resampling rate
                            win_length (int): window length for STFT
                            overlap (float): overlap between windows for STFT
        """
        self.directory = directory
        self.file_names = [f for f in os.listdir(self.directory)]

        # map file_names to participant IDs nad research stages
        self.uuid_to_id = (
            pd.read_csv('/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep/participant_id_to_uuid.csv')
            .assign(**{"Participant ID": lambda df: df["Participant ID"].astype(str)})
            .set_index("UUID")["Participant ID"]
            .to_dict())
        pheno_gold_data = pd.read_parquet('/net/mraid20/export/genie/LabData/Data/Pheno/gold/sleep/sleep_all.parquet'
                                            ).set_index('path_parquet_sleep_stage')
        pheno_gold_data.index = pheno_gold_data.index.str.split('/').str[5]
        pheno_gold_data = pheno_gold_data[pheno_gold_data.index.isin(self.file_names)]
        self.folder_to_uuid = pheno_gold_data['participant_uuid'].to_dict()
        self.folder_to_research_stage = pheno_gold_data['research_stage'].to_dict()

        self.ids = self._extract_ids(self.file_names)
        self.recordings = self._extract_recordings(self.file_names)
        self.output_format = output_format
        self.config = config

    def _extract_ids(self, file_names):
        """
        Extracts and returns unique IDs from file names.
        """
        mapped_ids = pd.Series(file_names).map(self.folder_to_uuid).map(self.uuid_to_id).tolist()
        unique_ids = list(set(mapped_ids))
        return unique_ids

    def _extract_recordings(self, file_names):
        """
        Extracts and returns unique recordings (in format ID_visitdate) from file names.
        """
        ids_and_dates = []
        for file_name in tqdm(file_names):
            if file_name in self.folder_to_uuid:
                person_id = self.uuid_to_id[self.folder_to_uuid[file_name]]
            else:
                person_id = 'unknown'
            # open the file parquet file and read the date
            dir = os.path.join(self.directory, file_name)
            night_dir = [f for f in os.listdir(dir) if f.startswith('night_')][0]
            parquet_file = [f for f in os.listdir(os.path.join(dir, night_dir)) if f.endswith('.parquet')][0]
            tmp = pd.read_parquet(os.path.join(self.directory, file_name, night_dir, parquet_file))
            recording_date = tmp.iloc[0, 2].date()
            # change date to format YYYYMMDD
            recording_date = recording_date.strftime('%Y%m%d')
            ids_and_dates.append(f"{person_id}__{recording_date}")
        unique_recordings = list(set(ids_and_dates))
        return unique_recordings

    def _get_id(self, file_name):
        if file_name in self.folder_to_uuid:
            person_id = self.uuid_to_id[self.folder_to_uuid[file_name]]
        else:
            person_id = 'unknown'
        return person_id

    def _get_research_stage(self, file_name):
        if file_name in self.folder_to_research_stage:
            research_stage = self.folder_to_research_stage[file_name]
        else:
            research_stage = 'unknown'
        return research_stage

    def _get_recording_date(self, file_name):
        # open the first parquet file and read the date
        dir = os.path.join(self.directory, file_name)
        night_dir = [f for f in os.listdir(dir) if f.startswith('night_')][0]
        parquet_file = [f for f in os.listdir(os.path.join(dir, night_dir)) if f.endswith('.parquet')][0]
        tmp = pd.read_parquet(os.path.join(self.directory, file_name, night_dir, parquet_file))
        recording_date = tmp.iloc[0, 2].date()
        # change date to format YYYYMMDD
        recording_date = recording_date.strftime('%Y%m%d')
        return recording_date

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # list all folder starting with "night=" in the visit directory
        recording_directory = os.path.join(self.directory, f'{self.file_names[idx]}')
        person_id = self._get_id(self.file_names[idx])
        recording_date = self._get_recording_date(self.file_names[idx])
        research_stage = self._get_research_stage(self.file_names[idx])
        night_dirs = [f for f in os.listdir(recording_directory) if f.startswith("night_")]
        biomarkers_df = {}
        for i, dir_name in enumerate(night_dirs):
            recording_name = f"{person_id}__{research_stage}__{i}"
            print(recording_name)

            # create a tensor for each night recorded and save it in PATH_FOR_PROCESSED_DATA
            night_directory = os.path.join(recording_directory, dir_name)
            tensor_data = []
            target_sr = self.config['target_sr']
            for signal_type in SIGNALS:
                tmp_file = os.path.join(night_directory, f'{signal_type}.parquet')
                if not os.path.exists(tmp_file):
                    break
                else:
                    df = pd.read_parquet(os.path.join(tmp_file))

                    # check if df contains at least 50% of non-NAN values and return (None, None) otherwise:
                    if df.isna().sum().sum() > 0.5 * df.size:
                        break

                    # check original sampling rate
                    delta_t = (df['time'][1] - df['time'][0])
                    if delta_t.seconds == 0:
                        sample_rate = 1 / (delta_t.microseconds / 10 ** 6)  # Hz
                    else:
                        sample_rate = 1 / delta_t.seconds

                    # if signal_type == 'pat_infra':
                    #     df = df['values'].to_numpy()
                    #     print(f"PPG dimensions : {df.shape}")
                    #
                    #     # compute pyPPG biomarkers
                    #     df, ppgSQI,  bm_defs, bm_vals, bm_stats = compute_pyppg_biomarkers(df, sample_rate, recording_name)
                    #     biomarkers_df.update({recording_name: {'pyppg_SQI': ppgSQI}})
                    #     for key in bm_stats.keys():
                    #         for bm in bm_stats[key].columns.to_list():
                    #             for stat in bm_stats[key].index.to_list():
                    #                 biomarkers_df[recording_name].update(
                    #                     {f'pyppg_{bm}_{stat}': bm_stats[key].loc[stat, bm]})
                    #
                    #     # convert back df to a dataframe for later merge into tensor_data
                    #     df = pd.DataFrame(df, columns=['values'])

                    # use unique sampling rate for all channels
                    if 0:  # for batched data
                        data = resample_batch_signal(df, fs_original=sample_rate, fs_target=target_sr, axis=-1)
                    else:
                        data = df['values'].interpolate().values.flatten()
                        if sample_rate > target_sr:
                            # downsample the data to get the desired target_sr using scipy.signal.decimate()
                            data = signal.decimate(data, int(sample_rate/target_sr), axis=0)
                        elif sample_rate < target_sr:
                            # interpolate the data to get higher resolution
                            data = np.interp(x=np.arange(0, len(data), sample_rate/target_sr), # xvals
                                             xp=np.arange(0, len(data)),  # x
                                             fp=data  # y
                                             )

                    # impute missing data (NaNs) with interpolation
                    data = pd.Series(data).interpolate().values

                    # append the data to the tensor_data list
                    tensor_data.append(torch.tensor(data.copy(), dtype=torch.float).unsqueeze(0))

            # count the number of signals in tensor_data and if less then 7 then continue to the next night
            if len(tensor_data) < len(SIGNALS):
                continue

            # count the number of NaN elements at the beginning of each tensor in the 2nd dimension, and keep the max
            max_nan_count = max([torch.isnan(tensor).sum(dim=1).squeeze(0) for tensor in tensor_data]).item()
            # clip the beginning of each tensor by max_nan_count elements
            tensor_data = [tensor[:, max_nan_count:] for tensor in tensor_data]
            # make sure they have same length in the 2nd dimension, if not - clip to the shortest one
            min_length = min([tensor.shape[1] for tensor in tensor_data])
            tensor_data = [tensor[:, :min_length] for tensor in tensor_data]
            # concatenate tensor_data on first dimension
            tensor_data = torch.cat(tensor_data, dim=0)
            duration = tensor_data.shape[1]/(target_sr*60*60)

            # if shorter than 5 hours return None
            if duration < self.config['min_duration']:
                continue

            # assert tensor_data does not contain any NaN value
            assert not torch.isnan(tensor_data).any()

            if self.output_format == 'timeseries':

                # save the preprocessed tensor with the same name as the original file recording name, in PATH_FOR_PROCESSED_DATA
                person_id = self.recordings[idx].split('__')[0]
                visit_date = self.recordings[idx].split('__')[1]
                mkdirifnotexists(PATH_FOR_PROCESSED_DATA)
                # torch.save(tensor_data, os.path.join(PATH_FOR_PROCESSED_DATA, f'preprocessed_{person_id}__{visit_date}__{i}.pt'))

                # segment the signal into smaller segments of size segment_length
                for sec_in_segment in self.config['sec_in_segment']:
                    segment_length = int(sec_in_segment * target_sr)
                    segmented_signal = []
                    for k in range(tensor_data.shape[0]):
                        segments = torch.cat([tensor_data[k, j:j + segment_length].unsqueeze(0) for j in
                                              range(0, tensor_data.shape[1], segment_length)][:-1], dim=0)
                        if segments.shape[0] > 0:
                            segmented_signal.append(torch.tensor(segments, dtype=torch.float).unsqueeze(0))
                    # concatenate the segments on dim=1
                    tmp = torch.cat(segmented_signal, dim=0).permute(1, 0, 2)
                    dir = PROJECT_DIR + f"preprocessed_data_SR_{RESAMPLE_RATE}Hz_{sec_in_segment}s_segments_gold"
                    mkdirifnotexists(dir)
                    torch.save(tmp, os.path.join(dir, f'preprocessed_{person_id}__{visit_date}__{i}.pt'))

            elif self.output_format == 'images':
                # transform each channel into a spectrogram image
                tensor_image = []
                tensor_image_log = []
                # window = signal.windows.blackman(self.config['win_length'])
                window = torch.ones(self.config['win_length'])
                for i in range(tensor_data.shape[0]):
                    f, t, Sxx = signal.spectrogram(tensor_data[i].numpy(),
                                                   fs=target_sr,
                                                   window=window,
                                                   nperseg=self.config['win_length'],
                                                   noverlap=int(self.config['overlap'] * self.config['win_length']))
                    tensor_image.append(torch.tensor(Sxx).unsqueeze(0))
                    tensor_image_log.append(torch.tensor(np.log(np.abs(Sxx) ** 2 + 1e-10), dtype=torch.float).unsqueeze(0))
                tensor_image = torch.cat(tensor_image, dim=0)
                tensor_image_log = torch.cat(tensor_image_log, dim=0)

                # display the spectrogram image and reconstruction of the signal
                if 0:
                    # plot the spectrogram image
                    fig, axs = plt.subplots(nrows=tensor_image.shape[0], figsize=(10, 15))
                    for i, ax in enumerate(axs):
                        ax.imshow(tensor_image_log[i].numpy(), extent=[0, 6, target_sr/2, 0], aspect='auto')
                        ax.set_title(SIGNALS[i], loc='right')
                        ax.set_ylabel('Frequency')
                        if i != tensor_image.shape[0] - 1:
                            ax.set_xticks([])
                    axs[0].set_title(
                            f'STFT window length of {int(self.config["win_length"] / target_sr)}sec - images of {tensor_image[0].shape}')
                    plt.xlabel('Time [hour]')
                    plt.tight_layout()
                    plt.savefig(os.path.join(PERSONAL_DIR, 'figures', f'spectrograms_{int(self.config["win_length"] / target_sr)}sec.png'))
                    plt.close()

                    # reconstruct the signals from spectrograms using inverse fourier transform
                    tensor_data_reconstructed = []
                    for i in range(tensor_image.shape[0]):
                        Sxx = tensor_image[i].numpy()
                        _, xrec = signal.istft(Sxx,
                                               fs=target_sr,
                                               window=window,
                                               nperseg=self.config['win_length'],
                                               noverlap=int(self.config['overlap'] * self.config['win_length']))
                        # make xrec the exact same size as the original signal
                        xrec = np.concatenate([xrec, np.zeros(tensor_data[i].shape[0] - xrec.shape[0])])
                        # normalize the reconstructed signal and the original signal
                        xrec = (xrec - xrec.mean()) / xrec.std()
                        tensor_data[i] = (tensor_data[i] - tensor_data[i].mean()) / tensor_data[i].std()
                        tensor_data_reconstructed.append(torch.tensor(xrec, dtype=torch.float).unsqueeze(0))
                    tensor_data_reconstructed = torch.cat(tensor_data_reconstructed, dim=0)

                    # compute the pearson correlation between the original and reconstructed signals
                    corr = torch.stack([torch.corrcoef(torch.stack([tensor_data[i], tensor_data_reconstructed[i]]))[0, 1]
                                        for i in range(tensor_data.shape[0])])
                    print(f'Pearson correlation between original and reconstructed signals: {corr}')
                    # wandb.log({f'Corr {SIGNALS[i]}': corr[i] for i in range(tensor_data.shape[0])})
                    # wandb.log({f'Mean Corr': corr.mean()})

                    # plot the original vs reconstructed signals
                    plt.figure(figsize=(10, 15))
                    for i in range(tensor_data.shape[0]):
                            plt.subplot(tensor_data.shape[0], 1, i + 1)
                            plt.plot(tensor_data[i].numpy(), label='original')
                            plt.plot(tensor_data_reconstructed[i].numpy(), label='inverse FFT of Spectogram')
                            plt.title(f'{SIGNALS[i]}      Pearson corr = {corr[i]:.2}', loc='right')
                            plt.xticks([])
                            plt.legend()
                    plt.xlabel('Time')
                    plt.tight_layout()
                    plt.savefig(os.path.join(PERSONAL_DIR, 'figures', f'original_vs_reconstructed_stft_{int(self.config["win_length"] / target_sr)}sec.png'))
                    plt.close()

                    # plot same figure but in zoomed-in to win_display sec
                    sec_to_display = 5*60 # 5 minutes
                    plt.figure(figsize=(10, 15))
                    for i in range(tensor_data.shape[0]):
                        plt.subplot(tensor_data.shape[0], 1, i + 1)
                        plt.plot(tensor_data[i].numpy(), label='original')
                        plt.plot(tensor_data_reconstructed[i].numpy(), label='inverse FFT of Spectogram')
                        plt.title(f'{SIGNALS[i]}      Pearson corr = {corr[i]:.2}', loc='right')
                        plt.xticks([])
                        plt.xlim([0, sec_to_display*target_sr])
                        # adjust the ylim to the variance in the zoomed-in window
                        plt.ylim([tensor_data[i].numpy()[:sec_to_display*target_sr].min(),
                                  tensor_data[i].numpy()[:sec_to_display*target_sr].max()])
                        plt.legend()
                    plt.xlabel('Time')
                    plt.tight_layout()
                    plt.savefig(os.path.join(PERSONAL_DIR, 'figures', f'original_vs_reconstructed_stft_{int(self.config["win_length"] / target_sr)}sec_zoomed.png'))
                    plt.close()

                # save the preprocessed tensor with the same name as the original file recording name, in PATH_FOR_PROCESSED_DATA
                person_id = self.recordings[idx].split('__')[0]
                visit_date = self.recordings[idx].split('__')[1]
                spec_dir = PATH_FOR_PROCESSED_DATA+f'_spec_win_{int(self.config["win_length"] / target_sr)}sec_overlap_{self.config["overlap"]}'
                mkdirifnotexists(spec_dir)
                torch.save(tensor_image_log, os.path.join(spec_dir, f'preprocessed_{person_id}__{visit_date}__{i}.pt'))
        #
        return biomarkers_df



    # compute biomarkers from pyPPG pipeline
def compute_pyppg_biomarkers(df, sample_rate, recording_name):
    """
    Computes pyPPG biomarkers from a single PPG signal.
    """
    try:
        from dotmap import DotMap
        from papagei.preprocessing.ppg import preprocess_one_ppg_signal
        from pyPPG import PPG, Fiducials
        import pyPPG.fiducials as FP
        import pyPPG.biomarkers as BM
        import pyPPG.ppg_sqi as SQI
    except ImportError as e:
        raise ImportError(
            "compute_pyppg_biomarkers requires pyPPG, dotmap, and papagei (preprocess_one_ppg_signal). "
            "Install e.g. `pip install pyPPG --no-deps` and `pip install dotmap` if needed. "
            f"Original import error: {e}"
        ) from e

    # from PyPPG
    df, ppg_d1, ppg_d2, ppg_d3 = preprocess_one_ppg_signal(waveform=df,
                                                           frequency=sample_rate)
    # save all pyPPG derivatives output (optional)

    signal = DotMap()
    signal.v = df
    signal.fs = sample_rate
    signal.filtering = True
    signal.fL = 0.5000001
    signal.fH = 12
    signal.order = 4
    signal.sm_wins = {'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}
    signal.ppg, signal.vpg, signal.apg, signal.jpg = df, ppg_d1, ppg_d2, ppg_d3

    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction = pd.DataFrame()
    correction.loc[0, corr_on] = True
    signal.correction = correction

    ## Create a PPG class
    s = PPG(s=signal, check_ppg_len=True)

    ## Get Fiducial points
    # Initialise the fiducials package
    fpex = FP.FpCollection(s=s)

    # Extract fiducial points
    fiducials = fpex.get_fiducials(s=s)
    if 0:
        # signal_for_pyppg = DotMap()
        # signal.v = waveform
        # signal.fs = frequency
        # signal.filtering = True
        dir = PYPPG_OUTPUT_DIR + recording_name
        mkdirifnotexists(dir)
        torch.save(torch.Tensor(df.copy()), os.path.join(dir, f'ppg_{recording_name}.pt'))
        torch.save(torch.Tensor(ppg_d1.copy()), os.path.join(dir, f'ppg_d1_{recording_name}.pt'))
        torch.save(torch.Tensor(ppg_d2.copy()), os.path.join(dir, f'ppg_d2_{recording_name}.pt'))
        torch.save(torch.Tensor(ppg_d3.copy()), os.path.join(dir, f'ppg_d3_{recording_name}_.pt'))


    # Create a fiducials class
    fp = Fiducials(fp=fiducials)

    ## PPG SQI
    try:
        ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
    except Exception:
        ppgSQI = np.nan

    ## Get Biomarkers and Statistics
    bm_defs = bm_vals = bm_stats = None
    if len(fiducials) > 0:
        # pyPPG may emit pandas NA values in fiducials (especially with pandas>=3),
        # which later fail inside biomarker slicing (`int(NA)`).
        fiducials_clean = fiducials.copy()
        if hasattr(fiducials_clean, "replace"):
            fiducials_clean = fiducials_clean.replace({pd.NA: np.nan})
        if hasattr(fiducials_clean, "apply"):
            fiducials_clean = fiducials_clean.apply(pd.to_numeric, errors="coerce")
        # Keep only rows with valid onset and systolic peak indices; biomarkers require both.
        # Coerce to float64: pyPPG/pandas may yield object columns; np.isfinite then raises TypeError.
        for req_col in ("on", "sp"):
            if req_col in fiducials_clean.columns:
                v = pd.to_numeric(fiducials_clean[req_col], errors="coerce")
                vals = np.asarray(v, dtype=np.float64)
                mask = np.isfinite(vals)
                fiducials_clean = fiducials_clean.loc[mask].copy()
        if len(fiducials_clean) > 0:
            fp = Fiducials(fp=fiducials_clean)
            bmex = BM.BmCollection(s=s, fp=fp)
            try:
                bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
            except (TypeError, ValueError):
                # Some recordings still produce invalid beat windows inside pyPPG; treat as no biomarkers.
                bm_defs = bm_vals = bm_stats = None

    return df, ppgSQI, bm_defs, bm_vals, bm_stats


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_seed(SEED)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    config = {'target_sr': RESAMPLE_RATE,
              'min_duration': MIN_DURATION,
              'win_length': WIN_LENGTH,
              'overlap': OVERLAP,
              'sec_in_segment': SEC_IN_SEGMENT} # seconds}

    # Create preprocessed datasets
    print(f"Start preprocessing dataset with sample rate of {RESAMPLE_RATE} Hz ")

    preprocessing = PreprocessSleepDataset(PATH_FOR_DATA, 'timeseries', config=config)
    durations = []
    biomarkers_df = {}

    if 0:  # run in queue
        from LabQueue.qp import qp as qp
        from LabUtils.addloglevels import sethandlers

        sethandlers(file_dir=PERSONAL_DIR)
        os.chdir(mkdirifnotexists(os.path.join(PERSONAL_DIR, 'Logs')))
        with qp(jobname='preprocess', q=['himem8.q', 'himem7.q'], max_r=len(preprocessing)) as q:
            q.startpermanentrun()
            res = {i: q.method(preprocessing.__getitem__, (i,)) for i in range(len(preprocessing))}
            q_res = {k: q.waitforresult(v) for k, v in res.items()}
            for key in q_res.keys():
                dict = q_res[key]
                for k in dict.keys():
                    if k in biomarkers_df.keys():
                        biomarkers_df[k].update(dict[k])
                    else:
                        biomarkers_df.update({k: dict[k]})

    else:
        res = {}
        for i in tqdm(range(len(preprocessing))):
            res[i] = preprocessing[i]
            for key in res.keys():
                dict = res[key]
                for k in dict.keys():
                    if k in biomarkers_df.keys():
                        biomarkers_df[k].update(dict[k])
                    else:
                        biomarkers_df.update({k: dict[k]})

    # save biomarkers
    biomarkers_df = pd.DataFrame.from_dict(biomarkers_df, orient='index')
    # biomarkers_df.to_parquet(
    #     os.path.join(f'/net/mraid20/export/jafar/SleepFM/pyPPG_output_gold/pyPPG_biomarkers_gold.parquet'))

    print(f"Done. \n{len(preprocessing)} visits were preprocessed."
          f"In total {len(os.listdir(PATH_FOR_PROCESSED_DATA))} nights were saved in {PATH_FOR_PROCESSED_DATA}.")

    if 0:
        # Plot one sample from the train dataset in a single figure, each subplot shows the signal included in each separate channel of the Tensor
        sample_data, id = preprocessing[0]
        plt.figure(figsize=(10, 15))  # Adjust figure size if needed
        mins_to_display = 5
        for i, label in enumerate(SIGNALS):
            plt.subplot(sample_data.shape[0], 1, i + 1)
            start_x = 170000
            end_x = 170000 + (10 * 60 * mins_to_display)  # 1 min
            displayed_data = sample_data[i].numpy().flatten()[start_x:end_x]
            plt.plot(range(start_x, end_x), displayed_data)
            plt.ylim([displayed_data.min(), displayed_data.max()])
            plt.xlim([start_x, end_x])
            plt.title(label, loc='right')
            plt.xticks([])
        plt.xlabel(f'{mins_to_display} minute(s)')
        plt.tight_layout()  # This ensures the subplots don't overlap
        plt.show()

