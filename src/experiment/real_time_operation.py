# Manage datasets
import numpy as np

import matplotlib.pyplot as plt
from math import ceil

# Manage file paths
from pathlib import Path
import os

# Model
import torch
from collections import deque, Counter

# Syncronization
from time import time, sleep

# External libraries
from src.utilities.pytrigno import TrignoEMG

# Own implementations
from src.models.classification_pipeline import SingleNet_CNN_LSTM_ATTENTION, SingleNet_CNN_LSTM, EMGStreamProcessor

# Data types
from typing import Tuple, Dict

#==================#
# Global variables #
#==================#
EMG_FREQ = 2000
EEG_FREQ = 125
RMS_FREQ = 40                   # 40 for 500 samples, 125 for 32 samples (window)

EEG_USEABLE_CHANNELS = [2, 3, 6, 7, 8, 9, 10, 11]

EMG_LOWCUT = 20
EMG_HIGHCUT = 450
EEG_LOWCUT = 0.5
EEG_HIGHCUT = 30

EEG_NUM_CH = len(EEG_USEABLE_CHANNELS)
EMG_NUM_CH = 3

RMS_SAMPLING_WINDOW = 500           # 500 samples - 250 ms                      32 samples - 16 ms                                       
RMS_WINDOW_STEPSIZE = 50            # 50 samples - 25 ms (90 % overlap)         16 samples - 8 ms (50 % overlap)

HAMPEL_WINDOWSIZE = 100 
HAMPEL_SIGMA = 2                   # Usually 2

SLIDING_WINDOW_SAMPLES = 1000
SLIDING_WINDOW_STEPSIZE = 200

EMG_SELECT_SENSORS = (0, 2)
EMG_SAMPLES_PER_READ = 200

EMG_CONFIG_DICT = {
    'rms_windowsize' : RMS_SAMPLING_WINDOW,
    'rms_stepsize' : RMS_WINDOW_STEPSIZE,
    'hampel_windowsize' : HAMPEL_WINDOWSIZE,
    'hampel_sigma' : HAMPEL_SIGMA,
    'hampel_plot_option' : [False, None],
    'include_EMG' : False
}

REJECT_CONFIG_DICT = {
    'EEG_epoch_rejection_tolerance' : 6,
    'EMG_epoch_rejection_tolerance' : 6,
    'EEG_ch_acceptance' : 0,
    'EMG_ch_acceptance' : 0
}

# Routine:
# 1. Load real-time data in circular queues
# 2. Sliding window to extract data
# 3. Preprocess data
# 4. Load into model

# NOTE hent fully release og block alt andet
class EMGRealTime:
    def __init__(self, config_dict : Dict, select_sensors : Tuple = (0, 2), samples_per_read : int = 200, units : str = 'mV'):
        self.config = config_dict
        self.EMG_ins = TrignoEMG(channel_range = select_sensors, samples_per_read = samples_per_read, units = units)

    def start_stream(self):
        print('EMG stream initilizing')
        self.EMG_ins.start()

        t0 = time()
        flush_buffer = True
 
        while flush_buffer:
            
            self.EMG_ins.read()
            
            tnow = time()
            if tnow > t0 + 5:
                flush_buffer = False
            
        print('EMG ready to GO!')
    
    def start_stream_for_Exp2ObjGraspProtocol(self):
        print('EMG stream initilizing')
        self.EMG_ins.start()
        junk_data = []

        t0 = time()
        flush_buffer = True
 
        while flush_buffer:
            
            junk_data.append(self.extract_data())
            
            tnow = time()
            if tnow > t0 + 3:
                flush_buffer = False
            
        print('EMG ready to GO!')

        junk_data = np.concatenate(junk_data, axis = 0)

        return junk_data

    def end_stream_for_Exp2ObjGraspProtocol(self):
        junk_data = []

        t0 = time()
        flush_buffer = True
 
        while flush_buffer:
            
            junk_data.append(self.extract_data())
            
            tnow = time()
            if tnow > t0 + 3:
                flush_buffer = False

        junk_data = np.concatenate(junk_data, axis = 0)

        print('EMG stream terminated')
        self.EMG_ins.stop()
        return junk_data

    def end_stream(self):
        print('EMG stream terminated')
        self.EMG_ins.stop()

    def extract_data(self):
        return np.transpose(self.EMG_ins.read())

class Buffer:
    def __init__(self, max_size : int, num_ch : int, window_size : int, step_size : int):
        # Parameters for circular buffer
        self.max_size = max_size
        self.num_ch = num_ch
        self.buffer = np.zeros((max_size, num_ch))
        self.current_size = 0
        self.write_idx = 0

        # Parameters for sliding window
        self.window_size = window_size
        self.step_size = step_size              # Step_size equal to window_size means no overlap, step_size < window_size means overlap, step_size > window_size means gap between windows
        self.read_idx = 0                       # Track pointer

    def add_data(self, data : np.ndarray):
        n_samples = data.shape[0]

        # Last position where there is new data
        end_idx = (self.write_idx + n_samples) % self.max_size

        if end_idx < self.write_idx:                                # Overwrite old data (wrap around)
            split = self.max_size - self.write_idx                  # Number of samples that can be written to end of buffer before wrapping around
            self.buffer[self.write_idx:] = data[:split]             # Write first part of new data to end of buffer
            self.buffer[:end_idx] = data[split:]                    # Write remaining new data to beginning of buffer
        else:                                                       # Update data (No wrap around needed)
            self.buffer[self.write_idx:end_idx] = data              # Write new data into buffer
        
        self.write_idx = end_idx
        self.current_size = min(self.current_size + n_samples, self.max_size)   # Update current size of buffer (cannot exceed max size)

    def get_window(self):
        if self.current_size < self.window_size:
            print('Not enough data in buffer to extract window')
            return None
        
        start = self.read_idx
        end = start + self.window_size

        if end <= self.max_size:
            window = self.buffer[start:end]                         # Extract window of data from buffer (no wrap around)
        else:
            window = np.vstack((self.buffer[start:], 
                                self.buffer[:end % self.max_size])) # Extract window of data from buffer (wrap around)    

        # Update read pointer
        self.read_idx = (self.read_idx + self.step_size) % self.max_size

        return window

class Model():
    def __init__(self, path_to_model : Path, num_motions = 2):
        self.path_dir = path_to_model
        self.model = None
        self.device = None

        self.initilize_model()

        self.pred_mapping = {}

        actions = ['Contract', 'Release']
        if num_motions == 2:
            target = ['Index', 'Thumb']
        elif num_motions == 7:
            target = ['Pinky', 'Ring', 'Middle', 'Index', 'Thumb', 'Pinch', 'Cylinder']
        elif num_motions == 3:
            target = ['Index', 'Thumb', 'Pinch']
        else:
            raise ValueError(f'num_motion invalid : {num_motions}')

        
        label_idx = 0
        for targ in target:
            for act in actions:
                self.pred_mapping[label_idx] = targ + ' ' + act
                label_idx += 1
        self.pred_mapping[label_idx] = 'Rest'

    def initilize_model(self):
        if not os.path.exists(self.path_dir):
            raise FileExistsError(self.path_dir)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #pin_memory = torch.cuda.is_available()

        #======================#
        # Load model arguments #
        #======================#
        checkpoint = torch.load(f = self.path_dir / "model.pth", map_location = self.device)
        model_args = checkpoint["model_args"]

        # self.model = SingleNet_CNN_LSTM_ATTENTION(**model_args)
        self.model = SingleNet_CNN_LSTM(**model_args)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)

        self.model.eval()

        print('Model initilized')
    
    def predict(self, input_data):

        inp = torch.tensor(input_data, dtype = torch.float32).to(self.device).unsqueeze(0)          # Match model dim (1, S, Ch)

        with torch.no_grad():
            logits, _, _ = self.model(inp)
            pred_idx = torch.argmax(logits, dim=1).item()
        
        pred_map = self.pred_mapping[pred_idx]
        
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0, pred_idx].item()

        return pred_map, confidence
    
class StateLogic():
    def __init__(self):
        self.state = 'rest'
        self.active_target = None   # e.g. INDEX, THUMB, PINCH
        self.active_action = None   # contract, release, rest

    def update(self, pred, avg_potential):

        if pred is None:
            return self.state, False
        
        parts = pred.split()
                
        if len(parts) != 2:         # pred don't have two parts. This is used for REST
            return self.state, False

        # Target is the limb type
        # Action is contract, release
        target, action = parts[0].lower(), parts[1].lower()

        # ======================
        # REST → ACTIVATE
        # ======================
        if self.state == "rest":
            if action == "contract":
                self.state = f"ACTIVATE_{target}"
                self.active_target = target
                self.active_action = action

        # ======================
        # ACTIVE → RETURN
        # ======================
        elif self.state.startswith("ACTIVATE") and avg_potential < 0:
            if action == "release": #and target == self.active_target:
                # self.state = f"RETURN_{target}"
                self.state = "rest"
                self.active_action = action

        # ======================
        # RETURN → REST
        # ======================
        '''
        # Issue. Action need to be release and target has the be the previous action
        # Where the state might be in rest. So it will not always reach rest
        # When having mujoco. The REST state will be, when simulation/exo has reached init position.
        '''
        # elif self.state.startswith("RETURN"):
        #     if action == "RELEASE" and target == self.active_target:
        #         self.state = "REST"
        #         self.active_target = None

        return self.state, True

class PredictionVoting:
    
    def __init__(self, window_size : int = 5, required_votes : int = 3):
        
        self.window_size = window_size
        self.required_votes = required_votes
        
        self.pred_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)

    # ==========================================
    # Add prediction and return voted decision
    # ==========================================
    # Confidence-weighted voting
    def update(self, prediction, confidence):

        # Add newest prediction
        self.pred_buffer.append(prediction)
        self.confidence_buffer.append(confidence)

        # Wait until enough predictions
        if len(self.pred_buffer) < self.window_size:
            return None

        # Count occurrences
        counts = Counter(self.pred_buffer)

        # Most common prediction
        voted_pred, voted_count = counts.most_common(1)[0]

        # Require majority agreement
        # if voted_count < self.required_votes:
        #     return voted_pred

        # --------------------------------------
        # ONLY confidences from voted class
        # --------------------------------------
        voted_confidences = [
            conf
            for pred, conf in zip(self.pred_buffer, self.confidence_buffer)
            if pred == voted_pred
        ]

        # Mean confidence of voted class
        mean_conf = np.mean(voted_confidences)

        if mean_conf > 0.9:
            return voted_pred

        return None

    # ==========================================
    # Optional helper
    # ==========================================
    def get_buffer(self):
        return list(self.pred_buffer)

    # ==========================================
    # Reset buffer
    # ==========================================
    def reset(self):
        self.pred_buffer.clear()

#========================#
# Mujoco implementaition #
#========================#
class State():
    def __init__(self):
        self.mode = 'REST'          # REST / ACTIVE / RETURN
        self.motion = None          # Index, thumb, ect.
        self.busy = False
        self.init_pose = True
        self.priority = 0
        self.motion_dict = {
        "index": {
            "joints": [7, 9],
            "priority": 1,
            "interrupt": True
        },  
        "thumb": {
            "joints": [3, 4, 5, 6], 
            "priority": 1,
            "interrupt": True
        },
        "middle": {
            "joints": [11, 13],
            "priority": 1,
            "interrupt": True
        },
        "ring": {
            "joints": [15, 17],
            "priority": 1,
            "interrupt": True
        },
        "pinky": {
            "joints": [19, 21],     
            "priority": 1,
            "interrupt": True
        },
        "pinch": {
            "joints": [3, 4, 9],    
            "priority": 1,
            "interrupt": True
        },
        "cylinder": {
            "joints": [3, 4, 5, 6, 9, 13, 17, 21],        
            "priority": 1,
            "interrupt": True
        }
        }
    
    def reset_priority(self):
        for motion_key in self.motion_dict:
            self.motion_dict[motion_key]['priority'] = 1

class SimulationStateLogic():
    def __init__(self):
        self.state = State()

    def update(self, pred, avg_potential, finished):
        '''
        Update prediciton state
        '''

        # If currently executing → ignore predictions
        if self.state.busy:

            self.state.init_pose = False

            if finished:
                self._on_finish()

            return self.state

        # Ignore low confidence
        if pred is None:
            return self.state
        
        parts = pred.split()          # e.g. "Index Contract"

        if len(parts) != 2:
            return self.state
        
        motion_name, action = parts[0].lower(), parts[1].lower()

        # Only allow transitions when NOT busy
        if self.state.mode == 'REST' and action == 'contract':
            self._start_motion(motion_name)
        
        elif self.state.mode == 'ACTIVE' and action == 'release' and avg_potential > 0:
            # Give high priority for the current motion
            if motion_name == self.state.motion:
                self._start_return()        
        
        elif self.state.mode == 'ACTIVE' and action == 'contract':
            self._force_motion(motion_name)

        return self.state
    
    def _start_motion(self, motion_name):
        self.state.mode = "ACTIVE"
        self.state.motion = motion_name.lower()
        self.state.busy = True
        # self.state.priority = self.motion_dict[motion_name]["priority"]

    def _start_return(self):
        self.state.mode = 'RETURN'
        self.state.busy = True

    def _on_finish(self):
        if self.state.mode == "ACTIVE":
            self.state.busy = False

        elif self.state.mode == "RETURN":
            self.state = State()   # back to REST
    
    def _force_motion(self, motion_name):
        self.state.mode = "ACTIVE"
        self.state.motion = motion_name
        self.state.busy = True
        # self.state.priority = self.motion_dict[motion_name]["priority"]

class MujocoPredictionVoting(PredictionVoting):
    def __init__(self,
                 window_size : int = 5,
                 required_votes : int = 3):

        super().__init__(
            window_size = window_size,
            required_votes = required_votes
        )

        self.priority_factor = 2
        self.priority_neutral = 1
        self.previous_motion = 'none'
        self.confidence_threshold = 0.9
        
        self.pred_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)

        self.motion_dict = {
        "index": {
            "joints": [7, 9],
            "priority": 1,
            "interrupt": True
        },  
        "thumb": {
            "joints": [3, 4, 5, 6], 
            "priority": 1,
            "interrupt": True
        },
        "middle": {
            "joints": [11, 13],
            "priority": 1,
            "interrupt": True
        },
        "ring": {
            "joints": [15, 17],
            "priority": 1,
            "interrupt": True
        },
        "pinky": {
            "joints": [19, 21],     
            "priority": 1,
            "interrupt": True
        },
        "pinch": {
            "joints": [3, 4, 9],    
            "priority": 1,
            "interrupt": True
        },
        "cylinder": {
            "joints": [3, 4, 5, 6, 9, 13, 17, 21],        
            "priority": 1,
            "interrupt": True
        }
        }

    # ==========================================
    # Add prediction and return voted decision
    # ==========================================
    # Confidence-weighted voting
    def update(self, prediction: str | None, confidence: float ):
    
        # Add newest prediction
        self.pred_buffer.append(prediction)
        self.confidence_buffer.append(confidence)

        # Wait until enough predictions
        if len(self.pred_buffer) < self.window_size:
            return None, 'none'

        counts = Counter(self.pred_buffer)

        # count_status = (predtion of motion , number of times its included)
        voted_pred, voted_count = counts.most_common(1)[0]
        
        if voted_pred == 'Rest':
            return 'rest', 'none'
        
        motion_name, action = voted_pred.split()
        motion_name = motion_name.lower()
        action = action.lower()
        rest_counter = 0

        if action == 'contract':
            # voted_confidences contain the confidence values of the predictions that match the voted_pred
            voted_confidences = [
            conf
            for pred, conf in zip(self.pred_buffer, self.confidence_buffer)
            if pred == voted_pred
            ]

            # Mean confidence of voted class
            mean_conf = np.mean(voted_confidences)

            if mean_conf > self.confidence_threshold:
                return voted_pred, 'contract'
            else:
                return None, 'none'

        elif action == 'release':
            majority_summary = {}
            for pred_motion_name, pred_confidence in zip(self.pred_buffer, self.confidence_buffer):     # Iterate over predictions and its confidence in the buffer
                parts = pred_motion_name.split()

                if len(parts) != 2:     # Account for Rest predictions in buffer
                    rest_counter += 1
                    motion_name = 'Rest'
                    if rest_counter >= self.required_votes:
                        return None, 'none'
                    continue
                elif len(parts) == 2:
                    motion_name, action = parts[0].lower(), parts[1].lower()
                else:
                    raise ValueError(f"Invalid prediction format: {pred_motion_name}")

                priority_value = self.motion_dict[motion_name]['priority']             # extract priority value
                score = pred_confidence * priority_value                                                # Compute score based on confidence and priority
                
                if motion_name in majority_summary.keys():
                    majority_summary[pred_motion_name]['score'] += score
                    majority_summary[pred_motion_name]['count'] += 1
                else:
                    majority_summary[pred_motion_name] = {
                        'score' : score,
                        'count' : 1
                    }

            # Extract the name of the highest score
            highscore_motion_name = max(
            majority_summary,
                key=lambda k: majority_summary[k]['score']
            )

            # Extract the dict value of highest score and count
            highscore_motion_value = majority_summary[highscore_motion_name]

            # Mean confidence of voted class
            mean_score = highscore_motion_value['score'] / highscore_motion_value['count']

            if mean_score > self.confidence_threshold:
                return highscore_motion_name, 'release'
            else:
                return None, 'none'

        return None, 'none'

class DemoSimulationStateLogic():
    def __init__(self):
        self.state = State()

    def update(self, pred, avg_potential):
        '''
        Update prediciton state
        '''

        # If currently executing → ignore predictions
        # if self.state.busy:

        #     self.state.init_pose = False

        #     if finished:
        #         self._on_finish()

        #     return self.state

        if self.state.mode == 'RETURN':
            self._on_finish()
            return self.state

        # Ignore low confidence and when buffer is not full
        if pred is None:                
            return self.state
        
        parts = pred.split()          # e.g. "Index Contract"

        if len(parts) != 2:           # For rest
            # if self.state.mode == 'RETURN' and pred == 'rest':      # Return to rest state
            #     self._on_finish()
            
            return self.state
        
        motion_name, action = parts[0].lower(), parts[1].lower()

        # Only allow transitions when NOT busy
        if self.state.mode == 'REST' and action == 'contract':
            self._start_motion(motion_name)
        
        elif self.state.mode == 'ACTIVE' and action == 'release' and avg_potential > 0:
            # Give high priority for the current motion
            if motion_name == self.state.motion:
                self._start_return()        
        
        elif self.state.mode == 'ACTIVE' and action == 'contract':
            self._force_motion(motion_name)

        return self.state
    
    def _start_motion(self, motion_name):
        self.state.mode = "ACTIVE"
        self.state.motion = motion_name.lower()
        self.state.busy = True
        # self.state.priority = self.motion_dict[motion_name]["priority"]

    def _start_return(self):
        self.state.mode = 'RETURN'
        self.state.busy = True

    def _on_finish(self):
        if self.state.mode == "ACTIVE":
            self.state.busy = False

        elif self.state.mode == "RETURN":
            self.state = State()   # back to REST
    
    def _force_motion(self, motion_name):
        self.state.mode = "ACTIVE"
        self.state.motion = motion_name
        self.state.busy = True
        # self.state.priority = self.motion_dict[motion_name]["priority"]


''' examine_latency
def examine_latency():
    base_dir = Path(__file__).resolve().parent / 'data'
    print(base_dir)
    find_files_ins = load_datasets(base_dir = base_dir)
    find_file = find_files_ins.find_flex_files
    subjects = [f'subject_{i}' for i in range(0, 2)]

    DATA = []
    for subj in subjects:
        data_path = find_file(subjects = subj,
                                modality = 'EMG',
                                fingers = 'index',
                                prefix = 'flex')
        
        subj_data = []
        
        for file in data_path:
            data = pd.read_csv(file).to_numpy()
            
            subj_data.append(data)
        
        DATA.append(np.concatenate(subj_data, axis = 0))
        
    
    DATA = np.concatenate(DATA, axis = 0)
    print(DATA.shape)

    EMG = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                      select_sensors = EMG_SELECT_SENSORS,
                      samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG_BUFFER = Buffer(max_size = 2000,
                        num_ch = EMG_NUM_CH,
                        window_size = RMS_SAMPLING_WINDOW,
                        step_size = RMS_WINDOW_STEPSIZE)
    
    t_buffer = []
    t_preprocess = []

    for chunk in range(0, DATA.shape[0], 500):
        
        chunk_data = DATA[chunk:chunk+500]          # Read data
        t0 = perf_counter()

        EMG_BUFFER.add_data(chunk_data)             # Load into circular buffer

        data_window = EMG_BUFFER.get_window()       # Extract window of data by sliding window

        t_buffer_temp = perf_counter()

        # data_clean = EMG.preprocess(data_window)                    # Preprocess window of data

        t_preprocess_temp = perf_counter()

        t_buffer.append(t_buffer_temp - t0)
        t_preprocess.append(t_preprocess_temp - t_buffer_temp)
    
    print(f'Average time for buffer operations: {np.mean(t_buffer) * 1000:.2f} ms')
    print(f'Average time for preprocessing: {np.mean(t_preprocess) * 1000:.2f} ms')'''

def Trigno_test():
    EMG = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                    select_sensors = (0, 1),
                    samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG.start_stream()

    print('Sleep')
    tim = 3
    sleep(tim)

    EMG.end_stream()

    data = EMG.extract_data()

    for i in range(5*tim + 5):
        print(i)
        data = np.array(data)
        print(data.shape)
        print(data.mean())

def load_exisiting_datasets(num_motions = 2, demo_motion = 'indexDemo'):
    
    from src.models.classification_pipeline import Manage3Split, SingleManageDataset
    from torch.utils.data import DataLoader

    TRIAL_PERIOD = 9 
    TRIM_PERIOD = 3
    SEED = 42
    data_dir = Path(__file__).resolve().parents[2] / 'mujoco/data'           #'data'    usual location
    print(data_dir)
    EMG_ins = EMGStreamProcessor(fs = EMG_FREQ, lowcut=EMG_LOWCUT, highcut=EMG_HIGHCUT,
                                 reject_config_dict = REJECT_CONFIG_DICT,
                                rms_window = 500, rms_step = 50,
                                hampel_window = 100, hampel_sigma = 2, base_dir = data_dir,
                                disable_rejection = True)     # Disable rejection to include all data for demo and analysis purposes. For real-time application, should enable rejection to ensure data quality
    split_ins = Manage3Split(seed = SEED)

    X_epoch = {}
    X_labels = {}
    print(demo_motion)
    if demo_motion == 'all':
        print('ENTER')
        motion_list = ['pinkyDemo', 'ringDemo', 'middleDemo', 'indexDemo', 'thumbDemo', 'pinchDemo', 'cylinderDemo']
        label_mapping = ['Pinky', 'Ring', 'Middle', 'Index', 'Thumb', 'Pinch', 'Cylinder'] 
        demo_iter = 0

    elif num_motions == 7:
        motion_list = ['pinky', 'ring', 'middle', 'index', 'thumb', 'pinchGrip', 'fullGrip']
    elif num_motions == 3:
        motion_list = ['index', 'thumb', 'pinchGrip']
    elif num_motions == 2:
        motion_list = ['index', 'thumb']
        
    
    SUBJECTS_IDs = ['subject_0']

    for subj in SUBJECTS_IDs:
        X_epoch[subj] = {}
        X_labels[subj] = {}

        for ml in motion_list:
            data, num_epochs = EMG_ins.load_subject_data(subj = subj, finger = ml, modality = 'EMG', trim_period = TRIM_PERIOD, trial_period = TRIAL_PERIOD)    

            if demo_motion == 'all':
                motion = label_mapping[demo_iter]
                demo_iter += 1
                X_epoch[subj][motion] = data

                
            else:
                # Trial-level split
                train_idx, val_idx, test_idx = split_ins._split_trials(num_trials = num_epochs, train_ratio = 0.7)

                if ml == 'pinchGrip':
                    motion = 'Pinch'
                elif ml == 'fullGrip':
                    motion = 'Cylinder'
                else:
                    motion = str.capitalize(ml)

                # Split each window into train, val, test sets
                X_epoch[subj][motion] = data[test_idx]

    return X_epoch
    
def main_previous_datasets(model_folder_name : str = 'SingleNet_CNN+LSTM+ATTENTION_EMG/subject_0', num_motions : int = 7, demo_motion : str = 'none', demo_start_data : int = 0, plot_EMG : bool = True):
    model_path_folder = Path(__file__).resolve().parents[1] / f"models/loggings/real_time/{model_folder_name}"

    MODEL = Model(path_to_model = model_path_folder, num_motions = num_motions)

    # STREAM = EMGRealTime(config_dict = EMG_CONFIG_DICT,
    #                   select_sensors = EMG_SELECT_SENSORS,
    #                   samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG_BUFFER = Buffer(max_size = 10000,
                        num_ch = EMG_NUM_CH,
                        window_size = SLIDING_WINDOW_SAMPLES,
                        step_size = SLIDING_WINDOW_STEPSIZE)
    
    PREPROCESS = EMGStreamProcessor(fs = EMG_FREQ, lowcut = EMG_LOWCUT, highcut = EMG_HIGHCUT,
                                    reject_config_dict = EMG_CONFIG_DICT, 
                                    rms_window = RMS_SAMPLING_WINDOW, rms_step = RMS_WINDOW_STEPSIZE,
                                    hampel_window = HAMPEL_WINDOWSIZE, hampel_sigma = HAMPEL_SIGMA,     # sigma usually 2
                                    base_dir = 'Unused')
    
    VOTER = MujocoPredictionVoting(window_size = 5, required_votes = 3)
    
    STATE = DemoSimulationStateLogic()
    
    X_epoch = load_exisiting_datasets(num_motions = num_motions, demo_motion = demo_motion)
    

    correct = 0
    total = 0

    mu = np.load(model_path_folder / "mu.npy")
    sigma = np.load(model_path_folder / "sigma.npy")

    
    # STREAM.start_stream()                      # Initilize streaming
    
    if num_motions == 3:
        motion_list = ['Index', 'Thumb', 'Pinch']
    elif num_motions == 7:
        motion_list = ['Pinky', 'Ring', 'Middle', 'Index', 'Thumb', 'Pinch', 'Cylinder']
    
    label_order = []
    for limb in motion_list:
        for act in ['Contract', 'Release']:
            comb = limb + ' ' + act
            label_order.append(comb)
    label_order.append('Rest')
    all_labels = []
    all_predictions = []

    #====================Plotting===================#
    num_channels = EMG_NUM_CH

    RMS_FREQ = EMG_FREQ / RMS_WINDOW_STEPSIZE      # 40 Hz
    WINDOW_SEC = 3                                 # show last 1 second
    BUFFER_LEN = int(RMS_FREQ * WINDOW_SEC)        # 40 samples

    if plot_EMG:
        plt.ion()

        n_rows = 3
        n_cols = ceil(num_channels / n_rows)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(10, 8),
            sharex=True,
            sharey=True
        )

        axes = axes.flatten()

        # x-axis from -1 sec -> 0 sec
        x = np.linspace(-WINDOW_SEC, 0, BUFFER_LEN)

        # Rolling visualization history
        history = np.zeros((BUFFER_LEN, num_channels))

        lines = []

        for ch in range(num_channels):

            ax = axes[ch]

            line, = ax.plot(x, history[:, ch], lw=1, color = 'orange')

            ax.set_title(f"Sensor {ch}")

            ax.set_xlim(-WINDOW_SEC, 0)
            ax.set_ylim(-5, 5)

            # IndexDemo -2, 2

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            lines.append(line)
            ax.grid(axis='y', linestyle='--', linewidth=0.3)
            ax.axhline(0, color='black', ls='-')

        # Hide unused axes
        for i in range(num_channels, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
        

    #=================================================================#
    time_tracker = []
    buffer_fill_size = 1
    WINDOW = 1000
    STEP = 200

    plot_iterations = 0
    demo_ranges = {
        'Index': slice(160*EMG_FREQ, None),
        'Middle': slice(53*EMG_FREQ, 105*EMG_FREQ),
        'Ring': slice(18*EMG_FREQ, None),
        'Pinky': slice(93*EMG_FREQ, None),
        'Pinch': slice(272*EMG_FREQ, None),
        'Cylinder': slice(77*EMG_FREQ, None),
        'Thumb': slice(30*EMG_FREQ, None),
    }
    demo_latency_dict = {}
    time_reset = [False, False]

    input('Wait for Enter')

    t0 = None
    t_total = time()
    try:
        for ml in motion_list:
            # X_emg = STREAM.extract_data()              # Read data        
            #X_emg, y_true = next(test_iter)
            
            #===================================================#
            # This is used for when using only one demo dataset #
            #===================================================#
            # if demo_motion != 'none':
                #motion_data = X_epoch['subject_0'][demo_motion]
                # motion_data = motion_data[demo_start_data:, :]
                # if not demo_single_execution:
                #     demo_single_execution = True
                # else:
                #     break

            # else:    
            #     motion_data = X_epoch['subject_0'][ml]
            #     motion_data = motion_data.reshape(-1, 3)
            # print('motion: ',motion_data.shape)

            if demo_motion != 'none' and num_motions == 7:
                motion_data = X_epoch['subject_0'][ml]
                motion_data = motion_data[demo_ranges[ml], :]
                demo_latency_dict[ml] = {}
                demo_latency_dict[ml]['ACTIVE'] = []
                demo_latency_dict[ml]['RETURN'] = []
                demo_single_execution = False
                print(motion_data.shape)

                if ml == 'Ring':
                    print('Break when Ring is reached')
                    break

            for start in range(0, motion_data.shape[0] - WINDOW + 1, STEP):

                if t0 is None:
                    t_total = time()
                    pass
                else:
                    while time() - t0 < 0.1:
                        pass
                X_emg = motion_data[start : start + STEP]
                
                t0 = time()
                
                # Load into circular buffer
                EMG_BUFFER.add_data(data = X_emg)
                
                if buffer_fill_size < 5:
                    buffer_fill_size += 1
                    continue
            
                plot_iterations += 1
                
                # Extract window of data by sliding window
                X_win = EMG_BUFFER.get_window()
                
                # Preprocess window of data
                X_pre = PREPROCESS.update(chunk = X_win)                # Without normalization

                if X_pre is None:
                    print('Pre is none')
                    continue
                
                # Normalize
                X_norm = (X_pre - mu) / (sigma + 1e-8)      
                X_avg_potential = np.mean(X_norm)      

                # Insert into model
                X_pred, confidence = MODEL.predict(input_data = X_norm)

                # Majority voting
                final_pred, pred_type = VOTER.update(X_pred, confidence)

                REST_THRESH = 0.003

                # Combine channels
                envelope = np.mean(X_pre, axis=1)

                # Activity magnitude
                activity = np.mean(envelope)

                # Rising/falling behavior
                slope = np.mean(np.diff(envelope))
                
                if activity < REST_THRESH:
                    y_true = 'Rest'  # rest

                elif slope > 0:
                    y_true = ml + ' ' + 'Contract'# contract
                    if not time_reset[0]:
                        time_reset[0] = True
                        print('Start contract', ml)
                else:
                    y_true = ml + ' ' + 'Release'
                    if not time_reset[1]:
                        time_reset[1] = True
                        print('Start release', ml)
                        print('\n')

                pred_correct = (X_pred == y_true)

                all_labels.append(y_true)
                all_predictions.append(X_pred)

                total += 1

                if pred_correct:
                    correct += 1
                
                # Output of the model
                state = STATE.update(pred = final_pred, avg_potential = X_avg_potential)
                
                # if new_state:
                # print(
                # f"STATE: {state.mode:<15} | "
                # f"PRED: {X_pred:<20} | "
                # f"FINAL: {final_pred} | "
                # f"RAW: {VOTER.get_buffer()} | ",
                # # f"CONF: {confidence:>6.2f}",
                # end="\r"
                # )

                if state.mode == 'REST':
                    pass
                elif state.mode == 'ACTIVE' and not demo_single_execution:         # demo_single_execution = False -> True
                    demo_latency_dict[ml]['ACTIVE'].append(time() - t0)
                    demo_single_execution = True

                elif state.mode == 'RETURN' and demo_single_execution:
                    demo_latency_dict[ml]['RETURN'].append(time() - t0)
                    demo_single_execution = False
                    time_reset[0], time_reset[1] = False, False

                time_diff = (time() - t0) * 1000
                time_tracker.append(time_diff)

                # if time_diff > 200:
                #     print('Time different is exceed - Prediction behind')

            # ===================================== #
            # Real-time plotting                    #
            # ===================================== #
                if plot_EMG:
                    N = X_norm.shape[0]

                    # If more samples than buffer size
                    if N > BUFFER_LEN:
                        X_norm = X_norm[-BUFFER_LEN:]
                        N = BUFFER_LEN

                    # Shift old samples left
                    history = np.roll(history, -N, axis=0)

                    # Insert newest samples
                    history[-N:, :] = X_norm

                    # Update each channel
                    for ch in range(num_channels):

                        y = history[:, ch]

                        lines[ch].set_ydata(y)

                        # # Dynamic y scaling
                        # ymin = np.min(y)
                        # ymax = np.max(y)

                        # margin = 0.1 * (ymax - ymin + 1e-8)

                        # axes[ch].set_ylim(
                        #     ymin - margin,
                        #     ymax + margin
                        # )

                    # Redraw
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                

    except KeyboardInterrupt:
        print('Terminate program')
    
    finally:
        # STREAM.end_stream()
        if plot_EMG:
            plt.ioff()
        # acc = correct / total * 100 if total > 0 else 0

        # print(f"\nFinal Accuracy: {acc:.2f}%")
        # print(f"Correct: {correct}/{total}")
        print(f"Total time: {time() - t_total:.2f} s")
        # from sklearn.metrics import confusion_matrix
        # import seaborn as sns

        # cm = confusion_matrix(
        #     all_labels,
        #     all_predictions,
        #     labels=label_order
        # )
        # print(cm)
        # cm_norm = cm / cm.sum(axis=1, keepdims=True)

        # plt.figure(figsize=(8, 6))

        # sns.heatmap(
        #     cm_norm,
        #     annot=True,
        #     fmt='.2f',
        #     cmap='Blues',
        #     xticklabels=label_order,
        #     yticklabels=label_order
        # )

        # plt.xlabel("Predicted")
        # plt.ylabel("True")
        # plt.title("Confusion Matrix")

        # plt.tight_layout()
        # plt.show()
        
        if len(time_tracker) > 0:
            print(f'Average time after extract data {np.mean(time_tracker):.2f} ms')
        

        print("Demo Latency:")
        for motion in demo_latency_dict:
            active_latencies = demo_latency_dict[motion]['ACTIVE']
            return_latencies = demo_latency_dict[motion]['RETURN']

            if len(active_latencies) > 0:
                print(active_latencies)
                avg_active_latency = np.mean(active_latencies) * 1000
                print(f"{motion} - ACTIVE latency: {avg_active_latency:.2f} ms")
            else:
                print(f"{motion} - ACTIVE latency: No detections")

            if len(return_latencies) > 0:
                print(return_latencies)
                avg_return_latency = np.mean(return_latencies) * 1000
                print(f"{motion} - RETURN latency: {avg_return_latency:.2f} ms")
            else:
                print(f"{motion} - RETURN latency: No detections")
            
def main_real_time(model_folder_name : str = 'SingleNet_CNN+LSTM+ATTENTION_EMG/subject_0', num_motions : int = 2, plot_real_time = False):
    model_path_folder = Path(__file__).resolve().parents[1] / f"models/loggings/real_time/{model_folder_name}"

    MODEL = Model(path_to_model = model_path_folder, num_motions = num_motions)

    STREAM = EMGRealTime(config_dict = EMG_CONFIG_DICT,
                      select_sensors = EMG_SELECT_SENSORS,
                      samples_per_read = EMG_SAMPLES_PER_READ)
    
    EMG_BUFFER = Buffer(max_size = 10000,
                        num_ch = EMG_NUM_CH,
                        window_size = SLIDING_WINDOW_SAMPLES,
                        step_size = SLIDING_WINDOW_STEPSIZE)
    
    PREPROCESS = EMGStreamProcessor(fs = EMG_FREQ, lowcut = EMG_LOWCUT, highcut = EMG_HIGHCUT,
                                    reject_config_dict = EMG_CONFIG_DICT, 
                                    rms_window = RMS_SAMPLING_WINDOW, rms_step = RMS_WINDOW_STEPSIZE,
                                    hampel_window = HAMPEL_WINDOWSIZE, hampel_sigma = HAMPEL_SIGMA,     # sigma usually 2
                                    base_dir = 'Unused')

    VOTER = PredictionVoting(window_size = 5, required_votes = 3)
    
    STATE = StateLogic()

    mu = np.load(model_path_folder / "mu.npy")
    sigma = np.load(model_path_folder / "sigma.npy")

    
    STREAM.start_stream()                      # Initilize streaming
    
    #====================Plotting===================#
    if plot_real_time:
        num_channels = EMG_NUM_CH

        RMS_FREQ = EMG_FREQ / RMS_WINDOW_STEPSIZE      # 40 Hz
        WINDOW_SEC = 1                                 # show last 1 second
        BUFFER_LEN = int(RMS_FREQ * WINDOW_SEC)        # 40 samples
        plt.ion()

        n_cols = 3
        n_rows = ceil(num_channels / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 6),
            sharex=True,
            sharey=True
        )

        axes = axes.flatten()

        # x-axis from -1 sec -> 0 sec
        x = np.linspace(-WINDOW_SEC, 0, BUFFER_LEN)

        # Rolling visualization history
        history = np.zeros((BUFFER_LEN, num_channels))

        lines = []

        for ch in range(num_channels):

            ax = axes[ch]

            line, = ax.plot(x, history[:, ch], lw=1)

            ax.set_title(f"Sensor {ch}")

            ax.set_xlim(-WINDOW_SEC, 0)
            ax.set_ylim(-5, 5)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            lines.append(line)

        # Hide unused axes
        for i in range(num_channels, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
        sleep(2)
        
    #=================================================================#
    time_tracker = []
    buffer_fill_size = 1

    try:
        while True:
            X_emg = STREAM.extract_data()              # Read data        
                
            t0 = time()
                
            # Load into circular buffer
            EMG_BUFFER.add_data(data = X_emg)
                
            if buffer_fill_size < 5:
                buffer_fill_size += 1
                continue
                
            # Extract window of data by sliding window
            X_win = EMG_BUFFER.get_window()
                
            # Preprocess window of data
            X_pre = PREPROCESS.update(chunk = X_win)                # Without normalization

            if X_pre is None:
                print('Pre is none')
                continue
                
            # Normalize
            X_norm = (X_pre - mu) / (sigma + 1e-8)        

            X_avg_potential = np.mean(X_norm)    

            # Insert into model
            X_pred, confidence = MODEL.predict(input_data = X_norm)

            # Majority voting
            final_pred = VOTER.update(X_pred, confidence)
            
            # Output of the model
            state, new_state = STATE.update(pred = final_pred, avg_potential = X_avg_potential)

            # if new_state:
            print(
            f"STATE: {state:<15} | "
            f"PRED: {X_pred:<20} | "
            f"FINAL: {final_pred} | "
            f"RAW: {VOTER.get_buffer()} | ",
            end="\r"
            )

            # ===================================== #
            # Real-time plotting                    #
            # ===================================== #
            if plot_real_time:
                N = X_norm.shape[0]

                # If more samples than buffer size
                if N > BUFFER_LEN:
                    X_norm = X_norm[-BUFFER_LEN:]
                    N = BUFFER_LEN

                # Shift old samples left
                history = np.roll(history, -N, axis=0)

                # Insert newest samples
                history[-N:, :] = X_norm

                # Update each channel
                for ch in range(num_channels):

                    y = history[:, ch]

                    lines[ch].set_ydata(y)


                # Redraw
                fig.canvas.draw_idle()
                plt.pause(0.001)
                
            time_diff = (time() - t0) * 1000
            time_tracker.append(time_diff)

            if time_diff > 200:
                print('Time different is exceed - Prediction behind')
            

    except KeyboardInterrupt:
        print('Terminate program')
    
    finally:
        STREAM.end_stream()
        plt.ioff()
        
        if len(time_tracker) > 0:
            print(f'Average time after extract data {np.mean(time_tracker):.2f} ms')



if __name__ == "__main__":

    # IndexDemo : START 160
    # MiddleDemo : START 53s and end 105s
    # ringDemo : START 18
    # pinkyDemo : START 93
    # pinchDemo : AROUND START 272s
    # cylinder : START 77s
    # thumb : START 30

    # model_folder_name = 'SingleNet_CNN+LSTM_EMG_complexModel_globalNorm_noWeight_3motions_lowpass/subject_0'
    model_folder_name = 'fine_tune/SingleNet_CNN+LSTM_EMG_newCrop_7motions/SingleNet_CNN+LSTM_EMG/subject_0'
    main_previous_datasets(model_folder_name = model_folder_name, num_motions = 7, demo_motion = 'all', demo_start_data = None, plot_EMG = True)
    # main_real_time(model_folder_name = model_folder_name, num_motions = 7, plot_real_time=True)



