# Communication
import socket

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
import time

# External libraries
from src.utilities.pytrigno import TrignoEMG

# Own implementations
from src.models.classification_pipeline import EMGStreamProcessor
from src.experiment.experimental_protocol import PROTOCOL_con
from src.experiment.real_time_operation import Model, EMGRealTime, Buffer, StateLogic, PredictionVoting
from src.utilities.esp_logger import ESPLogger

# Data types
from typing import Tuple, Dict

#==================#
# Global variables #
#==================#
EMG_FREQ = 2000
EEG_FREQ = 125
RMS_FREQ = 40                   # 40 for 500 samples, 125 for 32 samples (window)

EMG_LOWCUT = 20
EMG_HIGHCUT = 450
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

class Exp1BendFingerProtocol(PROTOCOL_con):
    def __init__(self,
                 num_epochs : int,
                 rest_duration : int,
                 onset_duration : int,
                 release_duration : int,
                 trim_duration : int = 3):
        
        super().__init__(
            num_epochs=num_epochs,
            rest_duration=rest_duration,
            onset_duration=onset_duration,
            release_duration=release_duration,
            trim_duration=trim_duration
        )

        # self.tcp_socket = None
        # self.ticks = (1494 * 3) - 1                   # 1494 ticks correspnds to one rotation

        # self.init_protocol(
        #     host = "10.126.128.82",  # <-- Replace with ESP IP from Serial Monitor
        #     port = 1234
        # )
        self.esp_host = "10.126.128.10"
        self.ESP = ESPLogger(host = self.esp_host, port = 1234)
        self.ESP.start_session(filename = 'ESP_log_test.csv')  

    def init_protocol(self, host, port):
        HOST = host  # <-- Replace with ESP IP from Serial Monitor
        PORT = port

        # Create TCP socket
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to ESP server
        self.tcp_socket.connect((HOST, PORT))

        # Confirm connection
        connected = False
        while not connected:

            self.tcp_socket.sendall(b'connect\n')

            scp_msg = self.tcp_socket.recv(1024).decode().strip()

            if scp_msg == 'connection complete':
                print('ESP connection established')
                connected = True
            else:
                print('Connecting...')
    
    def send_comment_esp(self, action : str):
        
        cmd = f"{action} {self.ticks}"
        
        try:
            msg = f"{cmd}\n".encode("utf-8")            # Raw bytes
            self.tcp_socket.sendall(msg)

        except Exception as e:
            print(f"ESP error: {e}")
    
    def execute_protocol(self, 
                         t0 : int,
                         epoch_idx : int,
                         file_handler,
                         send_queues_dict : dict):
        """Execute the experimental protocol.
        
        Args:
            num_epochs (int): Number of epochs to run
            rest_duration (float, optional): Duration of rest period in seconds. Defaults to 5.0.
            action_duration (float, optional): Duration of action period in seconds. Defaults to 5.0.
            release_duration (float, optional): Duration of release period in seconds. Defaults to 5.0.
            filepath (str, optional): Path to save markers. Defaults to None.
            barrier (multiprocessing.Barrier, optional): Synchronization barrier. Defaults to None.
        """
        print(f"Trial {epoch_idx}/{self.num_epochs}")

        # Add some control here. 
        #------------#
        # Rest event #
        #------------#
        # self.RES_SOUND.play()
        print('REST')
        t_epoch = time.perf_counter_ns()
        self.put_marker_to_queue(send_queues_dict = send_queues_dict, marker_id = self.REST_ID)
        self.log_marker(file_handler, self.diff(t0, t_epoch), marker_id = self.REST_ID, description = "Rest period started")
        t_wait = self.at(t_epoch, self.t_rest)
        self.wait_until(t_wait)

        #----------------#
        # Contract event #
        #----------------#
        # self.CON_SOUND.play()
        self.ESP.contract()
        print('Contract')
        # self.ESP.contract()
        self.put_marker_to_queue(send_queues_dict=send_queues_dict, marker_id = self.ONSET_ID)
        self.log_marker(file_handler, self.diff(t0, t_wait), marker_id = self.ONSET_ID, description = "Action period started")
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset)
        self.wait_until(t_wait)

        #---------------#
        # Release event #
        #---------------#
        # self.REL_SOUND.play()
        self.ESP.release()
        print('RELEASE')
        self.put_marker_to_queue(send_queues_dict=send_queues_dict, marker_id = self.REL_ID)
        self.log_marker(file_handler, self.diff(t0, t_wait), marker_id = self.REL_ID, description = "Release period started")
        t_wait = self.at(t_epoch, self.t_rest + self.t_onset + self.t_rel)
        self.wait_until(t_wait)

        if epoch_idx == self.num_epochs - 1:
            print("[OK] Experimental protocol completed.")
            self.ESP.end_session()
            return True

        return False

class Exp2ObjGraspProtocol(PROTOCOL_con):
    def __init__(self,
                 num_epochs : int,
                 rest_duration : int,
                 onset_duration : int,
                 release_duration : int,
                 trim_duration : int = 3):
        
        super().__init__(
            num_epochs=num_epochs,
            rest_duration=rest_duration,
            onset_duration=onset_duration,
            release_duration=release_duration,
            trim_duration=trim_duration
        )

        # self.tcp_socket = None
        # self.ticks = (1494 * 3) - 1                   # 1494 ticks correspnds to one rotation

        # self.init_protocol(
        #     host = "10.126.128.129",  # <-- Replace with ESP IP from Serial Monitor
        #     port = 1234
        # )

        self.control_periods_manager = [True, True, True, True]
        self.num_epochs = num_epochs
        self.ss = EMG_SELECT_SENSORS
        self.esp_host = "10.126.128.10"

        self.base_dir = Path().resolve() / 'src/experiment/data/metabolic_cost'
        exp_folder_name = 'demonstration'
        self.emg_log = self.base_dir / f'{exp_folder_name}/EMG/flex_test_finger.csv'
        self.esp_log = self.base_dir / f'{exp_folder_name}/ESP/flex_test_finger.csv'
        # self._create_file_header(filepath = self.emg_log)

    def init_protocol(self, host, port):
        HOST = host  # <-- Replace with ESP IP from Serial Monitor
        PORT = port

        # Create TCP socket
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to ESP server
        self.tcp_socket.connect((HOST, PORT))

        # Confirm connection
        connected = False
        while not connected:

            self.tcp_socket.sendall(b'connect\n')

            scp_msg = self.tcp_socket.recv(1024).decode().strip()

            if scp_msg == 'connection complete':
                print('ESP connection established')
                connected = True
            else:
                print('Connecting...')
    
    def _create_file_header(self, filepath):
        # if file doesn't exist, write header
        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")

        sensor_headers = [f'ch{i}' for i in range(self.ss[0], self.ss[1] + 1)]

        with open(filepath, 'w', newline='') as f:
            #np.savetxt(f, np.array(headers), delimiter=',', fmt='%s')
            f.write(','.join(sensor_headers) + '\n')
    
    def send_comment_esp(self, action : str):
        
        cmd = f"{action} {self.ticks}"
        
        try:
            msg = f"{cmd}\n".encode("utf-8")            # Raw bytes
            self.tcp_socket.sendall(msg)

        except Exception as e:
            print(f"ESP error: {e}")

    def execute_protocol(self, model_folder_name : str, num_motions : int):

        # file_handle = open(self.emg_log, 'a', buffering=1)
        
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

        ESP = ESPLogger(host = self.esp_host, port = 1234)
        
        STATE = StateLogic()

        mu = np.load(model_path_folder / "mu.npy")
        sigma = np.load(model_path_folder / "sigma.npy")

        trim_data = STREAM.start_stream_for_Exp2ObjGraspProtocol()
        # np.savetxt(file_handle, trim_data, delimiter=',', fmt='%.6f')
        
        buffer_fill_size = 1
        epoch_idx = 0
        t0 = time.time()
        
        loops = 0

        ESP.start_session(filename = self.esp_log)        

        try:
            while True:     # epoch_idx <= self.num_epochs - 1

                new_trial, action = self.control_periods(t0 = t0)

                loops += 1

                if new_trial:
                    print('AT THE END\n')
                    print('Time diff : ', time.time() - t0)
                    t0 = time.time()
                    epoch_idx += 1
                    loops = 0

                X_emg = STREAM.extract_data()              # Read data        

                # Save raw EMG
                # np.savetxt(file_handle, X_emg, delimiter=',', fmt='%.6f')
                    
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
                # final_pred = VOTER.update(X_pred, confidence)
                
                # Output of the model
                state, new_state = STATE.update(pred = X_pred, avg_potential = X_avg_potential)

                if new_state:

                    if  STATE.active_action == 'contract':          # action == 'contract':
                        # print('send contract')
                        # self.send_comment_esp('contract')                # Move motor
                        ESP.contract()
                    elif STATE.active_action == 'release':          # action == 'release':
                        # print('send contract')
                        # self.send_comment_esp('release')                # Move motor
                        ESP.release()
                
                
                print(
                f"STATE: {state:<15} | "
                f"PRED: {X_pred:<20} | ",
                # f"FINAL: {final_pred} | ",
                # f"RAW: {VOTER.get_buffer()} | ",
                end="\r"
                )

        except KeyboardInterrupt:
            print('Terminate program')
        
        finally:
            ESP.end_session()
            junk = STREAM.end_stream_for_Exp2ObjGraspProtocol()
            # np.savetxt(file_handle, junk, delimiter=',', fmt='%.6f')
    
    def control_periods(self,
                        t0):
        
        current_time = time.time()
        
        t_diff = current_time - t0
        
        # After t_rest say contract
        if t_diff >= 1 and self.control_periods_manager[0]:
            # print('CON SOUND', t_diff)
            self.CON_SOUND.play()
            self.control_periods_manager[0] = False
            return False, 'contract'
        
        # elif t_diff >= (1+5) and self.control_periods_manager[1]:
        #     # print('LIFT', t_diff)
        #     self.CON_SOUND.play()
        #     self.control_periods_manager[1] = False
        #     return False, 'contract'

        # elif t_diff >= (6+5) and self.control_periods_manager[2]:
        #     # print('LOWER', t_diff)
        #     self.REL_SOUND.play()
        #     self.control_periods_manager[2] = False
        #     return False, 'none'
    
        elif t_diff >= (1 + 5) and self.control_periods_manager[3]:
            # print('REL SOUND', t_diff)
            self.REL_SOUND.play()
            self.control_periods_manager[3] = False
            return False, 'release'

        elif t_diff >= (6 + 5):
            # print('FINISH', t_diff)
            self.control_periods_manager[0] = True
            self.control_periods_manager[1] = True
            self.control_periods_manager[2] = True
            self.control_periods_manager[3] = True
            return True, 'none'
        
        return False, 'none'

if __name__ == '__main__':
    
    PRO = Exp2ObjGraspProtocol(num_epochs = 50,
                                rest_duration = 1,
                                onset_duration = 6,
                                release_duration = 11,
                                trim_duration = 3)

    model_name = 'fine_tune/SingleNet_CNN+LSTM_EMG_newCrop_3motions/SingleNet_CNN+LSTM_EMG/subject_0'
    PRO.execute_protocol(model_folder_name = model_name,
                         num_motions = 3)


