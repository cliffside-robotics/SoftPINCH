from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.ndimage import median_filter
from pathlib import Path
import json
from scipy.ndimage import uniform_filter1d

class Filtering:
    def __init__(self, fs=125.0):
        self.fs = fs
        self.a = None
        self.b = None
        self.sos = None
        self.band_lowcut = None
        self.band_highcut = None

    def butter_bandpass(self, data, lowcut = 5, highcut = 40, order = 4):
        """
        Perform butterworth bandpass filtering. \n
        Perform notch before butter_bandpass \n
        args: \n
            EEG_data: EEG channel data - Dimension should be (samples, channels) \n
            lowcut: Define lowcut frequency \n
            highcut: Define highcut frequency \n
            order: Define order for the filter
        """
        self.band_lowcut = lowcut
        self.band_highcut = highcut
        # Create a Butterworth bandpass filter - b is the numerator coefficients, a is the denominator coefficients
        self.sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=self.fs, output='sos')     

        return signal.sosfiltfilt(self.sos, data, axis=0), self.sos

    def notch(self, data, cutoff = 50, Q = 30):
        """
        Perform Notch filtering. \n
        args: \n
            EEG_data: EEG channel data - Dimension should be (samples, channels) \n
            cutoff: Frequency to remove from the signal \n
            Q: Quality factor \n
                Higher Q → narrower notch (just kills a tiny band).
                Lower Q → wider notch (kills more around the center).
        """
        self.b, self.a = signal.iirnotch(w0=cutoff, Q=Q, fs=self.fs)
        return signal.filtfilt(self.b, self.a, data, axis=0)
    
    def lowpass_filter(self, data, cutoff = 5, order = 4):
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        sos = signal.butter(order, normal_cutoff, btype = 'low', output = 'sos')
        return signal.sosfiltfilt(sos, data, axis=0), sos
    
    def plot_visualize_filters(self, EEG_before_filtering, EEG_after_filtering, select_channel = 1, sos = None):
        """
        Figure 1: Visualize Butterworth bandpass filter characterization \n
        Figure 2: Visualize Power Spectral Density - That is dominant frequencies in the data \n
        args: \n
            Data before and after filtering - Dimension should be (samples, channels) \n
            select_channel: Which channel to display - Goes from 1 to max_channel
        """
        sel_ch = select_channel - 1

        # Figure 1
        plt.figure(figsize=(10, 5))
        w, h = signal.sosfreqz(sos, fs=self.fs)
        plt.semilogx(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
        plt.title(f'Butterworth BP {self.band_lowcut}-{self.band_highcut} Hz (fs={self.fs})')
        plt.xlabel('Frequency [Hz]'); plt.ylabel('Amplitude [dB]'); 
        plt.grid(True)
        #plt.xlim(-0.1, 100)
        plt.show()

        # Figure 2
        plt.figure(figsize=(10, 5))
        f_original, Pxx_original = signal.welch(EEG_before_filtering[:, sel_ch], fs=self.fs, nperseg=256)
        f_filtered, Pxx_filtered = signal.welch(EEG_after_filtering[:, sel_ch], fs=self.fs, nperseg=256)

        plt.semilogy(f_original, Pxx_original, label='Original Signal')
        plt.semilogy(f_filtered, Pxx_filtered, label='Filtered Signal')
        plt.title("Power Spectral Density Before and After Filtering")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_hampel_filter(self, original_signal: np.ndarray, filtered_signal: np.ndarray, outlier_indices: list, medians: np.ndarray, thresholds: np.ndarray, zoom : list | None = None):
        """
        Plot original and Hampel-filtered signals for all channels.

        Parameters
        ----------
        original_signal : np.ndarray
            Shape (n_samples, n_channels)
        filtered_signal : np.ndarray
            Shape (n_samples, n_channels)
        outlier_indices : list of lists
            outlier_indices[ch] contains indices for channel ch
        medians : np.ndarray
            Shape (n_samples, n_channels)
        thresholds : np.ndarray
            Shape (n_samples, n_channels)
        EMG_FREQ : float
            Sampling frequency (Hz)
        """
        if isinstance(zoom, list):
            st, ex = zoom
            st = st * self.fs
            ex = ex * self.fs
            original_signal = original_signal[st:ex, :]
            filtered_signal = filtered_signal[st:ex, :]
            medians = medians[st:ex, :]
            thresholds = thresholds[st:ex, :]

        n_samples, n_channels = original_signal.shape
        time = np.arange(n_samples) / self.fs
        ymax = np.max(original_signal)
        ymin = np.min(original_signal)

        # Create 2 subplots per channel
        fig, axes = plt.subplots(
            n_channels * 2,
            1,
            figsize=(14, n_channels * 8),
            sharex=True
        )

        # Ensure axes is always indexable
        if n_channels == 1:
            axes = np.array(axes)

        for ch in range(n_channels):

            ax_orig = axes[2 * ch]
            ax_filt = axes[2 * ch + 1]

            # ---------- Original signal ----------
            ax_orig.plot(
                time,
                original_signal[:, ch],
                linewidth=0.5,
                color="royalblue",
                label=f"Original (Ch {ch+1})"
            )

            ax_orig.fill_between(
                time,
                medians[:, ch] + thresholds[:, ch],
                medians[:, ch] - thresholds[:, ch],
                color="gray",
                alpha=0.3,
                label="Median ± Threshold"
            )

            # Mark detected outliers
            if len(outlier_indices[ch]) > 0 and zoom is None:
                ax_orig.plot(
                    time[outlier_indices[ch]],
                    original_signal[outlier_indices[ch], ch],
                    "ro",
                    markersize=0.5,
                    label="Outliers"
                )

            ax_orig.set_ylabel("Amplitude")
            ax_orig.set_title(f"Channel {ch+1} - Original with Hampel Bands")
            ax_orig.legend(loc="upper right")
            ax_orig.set_xlim(time[0], time[-1])
            ax_orig.set_ylim(ymin, ymax)

            # ---------- Filtered signal ----------
            ax_filt.plot(
                time,
                filtered_signal[:, ch],
                linewidth=0.5,
                color="seagreen",
                label=f"Filtered (Ch {ch+1})"
            )

            ax_filt.set_ylabel("Amplitude")
            ax_filt.set_title(f"Channel {ch+1} - Filtered")
            ax_filt.legend(loc="upper right")
            ax_filt.set_xlim(time[0], time[-1])
            ax_filt.set_ylim(ymin, ymax)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
    
    def zscore(self, data: np.ndarray, mode: str = "within_ch") -> np.ndarray:
        """
        Z-score standardization within or across channels.

        Parameters
        ----------
        data : np.ndarray
            Shape (samples,) or (samples, channels)
        mode : str
            'within_ch'  -> z-score independently per channel
            'across_ch'  -> z-score using global mean/std

        Returns
        -------
        np.ndarray
            Z-scored data with same shape as input
        """

        data = np.asarray(data)

        if mode == "within_ch":
            if data.ndim == 2:      # Z-score each channel independently when shape is (samples, channels)
                axis = 0          
            elif data.ndim == 3:
                axis = (0, 1)       # Z-score each channel independently when shape is (epoch, samples, channels)
        elif mode == "across_ch":
            axis = None
        else:
            raise ValueError("mode must be 'within_ch' or 'across_ch'")

        mean = np.mean(data, axis=axis, keepdims=True)
        std  = np.std(data, axis=axis, keepdims=True)
        std_max = np.maximum(std, 1e-8)

        return (data - mean) / (std_max)
    
    def hampel_filter(self, x: np.ndarray, window_size: int = 200, n_sigmas: float = 3.0, plot_filter_results: list = [False, None]):
        """
        Hampel filter for multi-channel signals.

        Parameters
        ----------
        x : np.ndarray
            Shape (n_samples, n_channels)
        window_size : int
            Number of samples on EACH side of the center sample
        n_sigmas : float
            Threshold multiplier
        plot_filter_results : list
            First element -> bool to display filter results
            second element -> list of specfic time window or None to display all

        Returns
        -------
        filtered_data   : np.ndarray (n_samples, n_channels)
        """

        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError("Input must be 2D: (samples, channels)")

        # Scale factor to make MAD comparable to standard deviation
        # (valid for approximately Gaussian data)
        k_scale = 1.4826
        kernel = 2 * window_size + 1

        # Median per channel (filter only along time axis)
        medians = median_filter(
            x,
            size=(kernel, 1),
            mode="reflect"
        )

        # Difference between real signal and the typical values of the signal
        diff = np.abs(x - medians)

        # Robust estimate of local variability (Median Absolute Deviation) per channel
        mad = k_scale * median_filter(
            diff,
            size=(kernel, 1),
            mode="reflect"
        )

        thresholds = n_sigmas * mad

        outlier_mask = diff > thresholds

        x_filt = x.copy()
        x_filt[outlier_mask] = medians[outlier_mask]

        if plot_filter_results[0]:
            # Collect outlier indices per channel
            outlier_indices = [
                np.nonzero(outlier_mask[:, ch])[0].tolist()
                for ch in range(x.shape[1])
            ]
            for ch in range(x.shape[1]):
                print('Number of outliers:', len(outlier_indices[ch])) 

            self.plot_hampel_filter(original_signal = x, filtered_signal = x_filt, outlier_indices = outlier_indices, medians = medians, thresholds = thresholds, zoom = plot_filter_results[1])
        
        return x_filt
    
class RejectBadEpochs():
    def __init__(self, base_dir : Path):
        '''
        Parameters
        ----------
        base_dir : Path
            Dictionary to where the data lies (Usualy: 'experiments/data')
        '''
        self.base_dir = base_dir

    def reject_routine(self,
                       data_file_per_finger : list,
                       epochs_overview : list,
                       EEG_data : np.ndarray | None = None,
                       RMS_data : np.ndarray | None = None,
                       reject_config_dict : dict = None,
                       EEG_useable_channels : list | None = None,
                       ) -> np.ndarray:
        '''
        Automatic and manual rejection of bad epochs. \n
        Parameters
        ----------
        data_file_per_finger : list
            List of paths to the files for each finger. Used for manual rejection.\n
            NOTE: Does not matter if EMG_files nor EEG_files. Use routine for different fingers

        epochs_overview : list
            List of number of epochs for each file. Used to convert continuous data into epochs and for manual rejection.

        EEG_data : np.ndarray
            Continuous EEG data after preprocessing. Shape (samples, channels)

        RMS_data : np.ndarray
            Continuous RMS data after preprocessing. Shape (samples, channels)

        reject_config_dict : dict
            Dictionary with parameters for the rejection routine.\n
            Should include keys:\n
            'EEG_epoch_rejection_tolerance' -> peak-to-peak auto-rejection tolerance.\n
            'EMG_epoch_rejection_tolerance' -> peak-to-peak auto-rejection tolerance.

        EEG_useable_channels : list | None = None
            List of ints, indicating which channels are of interest. 
            NOTE: Bad epochs at other channels are not considered in the final decision

        Returns
        ----------
        all_rejections_masks : np.ndarray of bools
            Boolean mask with shape (total_epochs,). True for bad epochs, False for good epochs.
        '''
        if isinstance(data_file_per_finger, Path):
            data_file_per_finger = [data_file_per_finger]
        if EEG_useable_channels is None:
            pass
        elif isinstance(EEG_useable_channels, list):
            EEG_data = EEG_data[:, EEG_useable_channels].copy()         # Extract data from selected channels
        else:
            raise ValueError(f'EEG_useable_channels is not of type list nor None. {type(EEG_useable_channels)}')
        #===================#
        # Convert to epochs #
        #===================#
        total_epochs = sum(epochs_overview)
        EEG_epoch = EEG_data.reshape(total_epochs, EEG_data.shape[0] // total_epochs, EEG_data.shape[1]) if EEG_data is not None else None
        RMS_epoch = RMS_data.reshape(total_epochs, RMS_data.shape[0] // total_epochs, RMS_data.shape[1]) if RMS_data is not None else None
        
        EEG_tolerance = reject_config_dict['EEG_epoch_rejection_tolerance']
        EMG_tolerance = reject_config_dict['EMG_epoch_rejection_tolerance']
        EEG_bad_ch_acceptance = reject_config_dict['EEG_ch_acceptance']
        EMG_bad_ch_acceptance = reject_config_dict['EMG_ch_acceptance']
        # Include number of bad channels before its bad

        st_epoch = 0
        all_rejections_masks = []
        
        for file, num_epoch in zip(data_file_per_finger, epochs_overview):
            #print(f"\nFile: {file}")
            EEG_autoreject = np.zeros(num_epoch, dtype=bool)   # Create np.ndarray of default false values corresponding to num_epochs size
            RMS_autoreject = np.zeros(num_epoch, dtype=bool)   # Create np.ndarray of default false values corresponding to num_epochs size
            
            if EEG_epoch is not None:
                EEG_autoreject = self.detect_bad_epochs_ptp(EEG_epoch[st_epoch : st_epoch + num_epoch], tolerance = EEG_tolerance, bad_ch_acceptance = EEG_bad_ch_acceptance)    
            if RMS_epoch is not None:
                RMS_autoreject = self.detect_bad_epochs_ptp(RMS_epoch[st_epoch : st_epoch + num_epoch], tolerance = EMG_tolerance, bad_ch_acceptance = EMG_bad_ch_acceptance)
            if EEG_epoch is None and RMS_epoch is None:
                raise ValueError('Input argument EEG_data and EMG_data is both none')

            # CALL MANUAL REJECTION FUNCTION WITH THE AUTO-REJECTED EPOCHS AS INPUT
            manualreject = self.detect_bad_epochs_manual(file, num_epoch)

            autorejct_combined = EEG_autoreject | RMS_autoreject | manualreject

            all_rejections_masks.append(autorejct_combined)
            st_epoch += num_epoch
        
        all_rejections_masks = np.concatenate(all_rejections_masks, axis = 0)       # Concat along epoch axis
        print('\n=====FUNC : reject_routine =====\n')
        print(f'Final combined bad epoch indicies: {np.where(all_rejections_masks)[0]}')
        print(f'Total bad epochs = {np.sum(all_rejections_masks)} out of {total_epochs}')

        return all_rejections_masks
    
    def detect_bad_epochs_ptp(self, data : np.ndarray, tolerance : int = 6, bad_ch_acceptance = 1) -> np.ndarray:
        """
        Detect bad epochs based on peak-to-peak amplitude.
        
        Parameters
        -----------
        data : np.ndarray 
            Epoch data with shape (n_epochs, n_times, n_channels)
        tolerance : int 
            Threshold multiplier for MAD. Higher k allows more tolerance

        Returns
        -----------
        bad_mask : np.ndarray of bools 
            A boolean mask with shape (n_epochs,). True for bad epochs, False for good epochs.
        """
        # Peak-to-peak per epoch per channel
        ptp = np.ptp(data, axis=1)  # (epochs, channels)

        # Robust threshold per channel
        med = np.median(ptp, axis = 0)                    # (c_channels,)
        mad = np.median(np.abs(ptp - med), axis = 0)      # (n_channels,) -> Purpose: Median Absolute Deviation (MAD) is a robust measure of variability that is less sensitive to outliers than standard deviation. It is calculated as the median of the absolute deviations from the median of the data. In this context, it provides a robust estimate of the variability in peak-to-peak values across epochs for each channel, which can be used to set a threshold for identifying bad epochs.

        # Avoid zero MAD
        mad[mad == 0] = 1e-12

        # Define threshold for for each channels
        threshold = med + tolerance * mad

        # Epoch is bad if ANY channel exceeds its threshold. 
        bad_mask = np.any(ptp > threshold, axis=1)

        bad_indices = np.where(bad_mask)[0]
        for idx in bad_indices:
            bad_ch_sum = np.where(ptp[idx] > threshold)[0]
            bad_ch_sum = bad_ch_sum.shape[0]
            
            if bad_ch_sum <= bad_ch_acceptance:
                bad_mask[idx] = False

        return bad_mask
    
    def detect_bad_epochs_manual(self, file : Path, num_epochs : int) -> np.ndarray:
        '''
        Detect bad epochs manually based on a JSON file.
        
        Parameters
        ----------
        file : Path
            Path to the current file being processed
        num_epochs : int
            Number of epochs in the current file

        Returns
        ----------
        bad_manual : np.ndarray of bools
            Boolean mask indicating which epochs are manually marked as bad
        '''
        bad_epoch_files = self.base_dir / 'manual_bad_epochs.json'
        
        if not bad_epoch_files.exists():
            raise FileNotFoundError(f'Manual bad epoch file not found at {bad_epoch_files}')
        
        subject = file.parents[1].name
        filename = file.stem
        key = f'{subject}_{filename}'
        

        with open(bad_epoch_files, 'r') as f:
            manual_json = json.load(f)                      # Load json file

            manual = manual_json[key]                       # Search for key in json file
            bad_manual = np.zeros(num_epochs, dtype=bool)   # Create np.ndarray of default false values corresponding to num_epochs size
        
            bad_manual[manual] = True                       # Convert indicies to True if mentioned in json file 
        
        return bad_manual

class EEG_preprocessing(Filtering):
    def __init__(self, fs = 125,
                 bandpass_lowcut : int = 2,
                 bandpass_highcut : int = 32,
                 trial_period : int = 9,
                 trim_period : int = 3):
        super().__init__()
        
        self.channel_names = [
        "Fp1", "Fp2",   # frontal pole
        "C3",  "C4",    # central
        "T5",  "T6",    # temporal (posterior)
        "Cz",  "Fz",    # occipital
        "F7",  "F8",    # temporal (anterior)
        "F3",  "F4",    # frontal
        "T3",  "T4",    # temporal (mid)
        "P3",  "P4"     # parietal
        ]
        self.fs = fs
        self.lowcut = bandpass_lowcut
        self.highcut = bandpass_highcut
        self.trial_period = trial_period
        self.trim_period = trim_period
        self.expected_num_epochs = 30

    def trim_trial_periods_UNUSED(self,
                           EEG_bandpass : np.ndarray,
                           all_markers: np.ndarray,
                           markers_idx: np.ndarray,
                           extract_event : str = 'ALL') -> tuple[np.ndarray, int]:
        """
        Trims the trial periods from the markers array.

        Parameters
        ----------
        EEG_bandpass : np.ndarray 
            Input of EEG data bandpass filtered
        all_markers : np.ndarray 
            Input of all markers array, including zero values
        markers_idx : np.ndarray
            Indices of non-zero markers. Can be extracted using np.nonzero()[0]
        extract_event : str
            Choise which segment of data to extract (For example: 'ALL', 'CONTRACT', 'RELEASE', 'REST')

        Returns
        ----------
        :return: List of trimmed segments
        :return: Int of the total amount of epochs for one experiment
        """
        if extract_event == 'ALL':
            START_TRIAL_MARKER = 10
            END_TRIAL_MARKER = {10, 201}
            TRIAL_PERIOD = 9
        elif extract_event == 'REST':
            START_TRIAL_MARKER = 10
            END_TRIAL_MARKER = {20, 201}
            TRIAL_PERIOD = 3
        elif extract_event == 'CONTRACT':
            START_TRIAL_MARKER = 20
            END_TRIAL_MARKER = {30, 201}
            TRIAL_PERIOD = 3
        elif extract_event == 'RELEASE':
            START_TRIAL_MARKER = 30
            END_TRIAL_MARKER = {10, 201}
            TRIAL_PERIOD = 3
        else:
            raise ValueError(f'{extract_event} is not valid event type')

        START_TRIM_MARKER = 101
        END_TRIM_MARKER = 102

        enter_trial = False             # Controls when to enter a new trial period
        enter_trim = False              # Controls when to enter a new trim period
        period_extrated = False         # Controls when the boundaries of an entire trial is selected
        num_epochs = 0                  # Increment each time a trial is appended
        st = 0                          # Start boundary index
        ed = 0                          # End boundary index
        data = EEG_bandpass.copy()      # Data
        resampled_data = []             # Resampled data to fit with desired frequency

        # Loop over non-zero indices where markers are present
        for i in range(len(markers_idx)):
            
            # Get the actual index in the markers array plus the next index
            mark_idx = markers_idx[i]
            mark_idx_next = markers_idx[i+1] if i + 1 < len(markers_idx) else None
            
            # Prevent iteration to go out of bounds. RELEASE require END_MARKER to trim the last trial
            if mark_idx_next is None and extract_event != 'RELEASE':
                continue

            # Enter only at the start of the first trim period
            elif all_markers[mark_idx] == START_TRIM_MARKER and not enter_trim:
                st = mark_idx
                enter_trim = True
            
            # Enter only in the end of the first trim period
            elif all_markers[mark_idx] == END_TRIM_MARKER and enter_trim:
                ed = mark_idx
                period_extrated = True
            
            # Enter only at the start of each trial period
            elif all_markers[mark_idx] == START_TRIAL_MARKER and not enter_trial:
                st = mark_idx
                enter_trial = True

            # Enter only at the end of each trial period when the START and END marker is different.
            elif all_markers[mark_idx] in END_TRIAL_MARKER and enter_trial:
                ed = mark_idx - 1
                period_extrated = True
            
            # Enter only in the end of each trial period, if START and END marker belong to the same mark
            elif all_markers[mark_idx_next] in END_TRIAL_MARKER and enter_trial and START_TRIAL_MARKER in END_TRIAL_MARKER:
                ed = mark_idx_next - 1
                period_extrated = True
                
            if period_extrated:
                period_extrated = False
                
                if enter_trim:
                    trim_data = data[st:ed, :]
                    n_samples = trim_data.shape[0]
                    real_fs = n_samples / TRIAL_PERIOD
                    target_len = int(np.round(n_samples * (self.fs / real_fs)))

                    enter_trim = False
                    #print(f'Trial from {st} to {ed}, len: {ed - st}')
                    #print(f'Time period {trim_period}\n Real fs: {real_fs} Hz\n Target len: {target_len}\n Original len: {n_samples}')
                     
                elif enter_trial:
                    trim_data = data[st:ed, :]
                    n_samples = trim_data.shape[0]
                    real_fs = n_samples / TRIAL_PERIOD
                    target_len = int(np.round(n_samples * (self.fs / real_fs)))

                    EEG_resampled = resample(trim_data, target_len, axis=0)
                    resampled_data.append(EEG_resampled)                  
                    
                    enter_trial = False
                    num_epochs += 1             
                    #print(f'Trial from {st} to {ed}, len: {ed - st}')
                    #print(f'Time period 9\n Real fs: {real_fs} Hz\n Target len: {target_len}\n Original len: {n_samples}')           
        
        return np.concatenate(resampled_data, axis=0), num_epochs
    
    def preprocessing_routine(self, raw_eeg : np.ndarray) -> tuple[np.ndarray, int]:
        '''
        Performs the full preprocessing routine:
        1) Notch + Bandpass filter
        2) Resample + z-score standardization + Secmentation into epochs

        Parameters
        ----------
        raw_eeg : np.ndarray
            This holds keys for a specfic class (finger). NOTE - If raw_eeg is a list, it will be converted to a dict with key 'single_class'. 2D array - Dim(samples, channels)

        Return
        ------
        :return: np.ndarray of normalized EEG data
        :return: Int of the total amount of epochs for one experiment
        '''
        # ---------------------------#
        # 1) NOTCH + BANDPASS FILTER #
        # ---------------------------#
        EEG_filter_ins = Filtering(fs = self.fs)
        
        EEG_notch = EEG_filter_ins.notch(data = raw_eeg, cutoff = 50, Q = 30)
        EEG_bandpass, _ = EEG_filter_ins.butter_bandpass(data = EEG_notch, lowcut = self.lowcut, highcut = self.highcut, order = 4)

        #===============================#
        # 2) Calculate number of epochs #
        #===============================#
        trim_samples = self.fs * self.trim_period
        samples_per_epoch = self.fs * self.trial_period

        valid_samples = EEG_bandpass.shape[0] - 2 * trim_samples            # Total samples for experimental period. WHY *2 : Trim egde on both sides
        num_epochs = int( np.round(valid_samples / samples_per_epoch) )     # Divide out total samples in sections of samples per epoch -> Results in number of epochs

        trim_start = trim_samples
        trim_end = trim_start + num_epochs * samples_per_epoch              # WHY instead of data[trim : -trim] -> Inconsistency in protocol causes the last batch of data not be included -> Rare but can happen

        if (trim_end - trim_start) % samples_per_epoch != 0:                # Inform if epochs is differnet from usual amount. Can happen if bad trials is removed.
            print(f"Warning: Samples not perfectly divisible by trial period. Calculated num epochs: {valid_samples / samples_per_epoch}")
            print(f'Trim samples at start and end: {trim_start}, {trim_end}\n')
            print(f"Total samples: {EEG_bandpass.shape[0]}, Valid samples: {valid_samples}, Samples per epoch: {samples_per_epoch}, Calculated num epochs: {num_epochs}")
        
        #=========#
        # 3) TRIM #
        #=========#       
        EEG_trim = EEG_bandpass[trim_start : trim_end, :]
        # print(f"Original shape {EEG_bandpass.shape}\n"
        #       f'EEG_trim shape: {EEG_trim.shape}\n')

        return EEG_trim, num_epochs
    
    def reject_channel_UNUSED(self, signal, print_rej_ch=False):
        mean_uV = np.mean(np.abs(signal), axis=0)
        std_uV = np.std(signal, axis=0)

        bad_mean = mean_uV > (np.mean(mean_uV) + 2*np.std(mean_uV))
        bad_std  = std_uV > (np.mean(std_uV) + 2*np.std(std_uV))
        keep_idx = np.where(~(bad_mean | bad_std))[0]
        
        if print_rej_ch:
            reject_idx = np.where((bad_mean | bad_std))[0]
            for i in reject_idx:
                print(f"Channel which is rejected: {self.channel_names[i]}")

        return signal[:, keep_idx], keep_idx

    def sliding_rms(self, signal, window_size=10, step_size=5):
        """
        Parameters
        ----------
        signal : np.ndarray
            Shape (epochs, sequence, channels)

        Returns
        -------
        rms : np.ndarray
            Shape (epochs, windows, channels)
        """

        E, T, C = signal.shape

        # Number of windows
        n_windows = (T - window_size) // step_size + 1

        rms = np.zeros((E, n_windows, C))

        for epo in range(E):
            for ch in range(C):
                w_idx = 0
                for start in range(0, T - window_size + 1, step_size):
                    window = signal[epo, start:start + window_size, ch]
                    rms[epo, w_idx, ch] = np.sqrt(np.mean(window**2))
                    w_idx += 1

        return rms
    
class EMG_preprocessing(Filtering):
    def __init__(self,
                 fs = 2000,
                 bandpass_lowcut : int = 20,
                 bandpass_highcut : int = 450,
                 trial_period : int = 9,
                 trim_period : int = 3,):
        super().__init__()

        self.emg_ch_names = [
        'Channel 1 : Palmaris longus',
        'Channel 2 : Flexor digitorum superficialis',
        'Channel 3 : Flexor pollicis longus',
        ]
        self.fs = fs
        self.lowcut = bandpass_lowcut
        self.highcut = bandpass_highcut
        self.trial_period = trial_period
        self.trim_period = trim_period
        self.expected_num_epochs = 30

    def sliding_rms(self, signal, window_size=10, step_size=5):
        rms_vals_all = []
        for ch in range(signal.shape[1]):
            rms_vals = []
            for start in range(0, len(signal) - window_size + 1, step_size):
                window = signal[start : start + window_size, ch]
                rms_vals.append(np.sqrt(np.mean(window**2)))
            rms_vals_all.append(rms_vals)
        return np.array(rms_vals_all).T

    def rms_conv(self, signal, window_size=32, step_size=16):
        '''
        Convolution RMS. RMS = sqrt( LPF(x^2) ), where LPF is implemented as a uniform filter (moving average) over the squared signal.
        '''
        power = signal**2

        mean_power = uniform_filter1d(
            power,
            size=window_size,
            axis=0,
            mode="nearest"
        )

        rms = np.sqrt(mean_power)

        return rms[::step_size]
    
    def preprocessing_routine(self,
                              raw_emg : np.ndarray,
                              rms_windowsize : int = 200,
                              rms_stepsize : int = 50,
                              hampel_windowsize : int = 200,
                              hampel_sigma : int = 3.0,
                              hampel_plot_option : list = [False, None]) -> tuple[np.ndarray, np.ndarray, int]:
        '''
        Performs the full preprocessing routine:
        1) Notch + Bandpass filter
        2) Resample + z-score standardization + Secmentation into epochs

        :param dict raw_emg: This holds keys for a specfic class (finger). NOTE - If raw_emg is a list, it will be converted to a dict with key 'single_class'. 2D array - Dim(samples, channels)
        :param int bandpass_lowcut: Lowpass frequency
        :param int bandpass_highcut: Highpass frequency
        :param int num_channels: Number of activated channels
        :param int num_epochs: Number of epochs (trials) in the dataset
        :param int trial_period: Time of each epoch

        :return np.ndarray EMG: Normalized continuous EMG data - EMG(samples, channels)
        '''
        # ---------------------------#
        # 1) NOTCH + BANDPASS FILTER #
        # ---------------------------#
        EMG_filter_ins = Filtering(fs = self.fs)        
        
        EMG_notch = EMG_filter_ins.notch(raw_emg, cutoff=50, Q=30)
        EMG_bandpass, _ = EMG_filter_ins.butter_bandpass(EMG_notch, lowcut = self.lowcut, highcut = self.highcut, order=4)

        #===============================#
        # 2) Calculate number of epochs #
        #===============================#
        trim_samples = self.fs * self.trim_period
        samples_per_epoch = self.fs * self.trial_period

        valid_samples = EMG_bandpass.shape[0] - 2 * trim_samples            # Total samples for experimental period. WHY *2 : Trim egde on both sides
        num_epochs = int( np.round(valid_samples / samples_per_epoch) )     # Divide out total samples in sections of samples per epoch -> Results in number of epochs

        trim_start = trim_samples
        trim_end = trim_start + num_epochs * samples_per_epoch              # WHY instead of data[trim : -trim] -> Inconsistency in protocol causes the last batch of data not be included -> Rare but can happen

        if (trim_end - trim_start) % samples_per_epoch != 0:                # Inform if epochs is differnet from usual amount. Can happen if bad trials is removed.
            print(f"Warning: Samples not perfectly divisible by trial period. Calculated num epochs: {valid_samples / samples_per_epoch}")
            print(f'Trim samples at start and end: {trim_start}, {trim_end}\n')
            print(f"Total samples: {EMG_bandpass.shape[0]}, Valid samples: {valid_samples}, Samples per epoch: {samples_per_epoch}, Calculated num epochs: {num_epochs}")

        #=========#
        # 3) TRIM #
        #=========#       
        EMG_trim = EMG_bandpass[trim_start : trim_end, :]
        
        # -----------------#
        # 4) Hampel filter #
        # -----------------#
        EMG_hampel = EMG_filter_ins.hampel_filter(x = EMG_trim, window_size = hampel_windowsize, n_sigmas = hampel_sigma, plot_filter_results = hampel_plot_option)

        # -------#
        # 5) RMS #
        # -------#
        RMS = self.rms_conv(signal = EMG_hampel, window_size = rms_windowsize, step_size = rms_stepsize)
        
        return RMS, EMG_hampel, num_epochs

if '__main__' == __name__:
    pass