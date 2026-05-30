from ..utilities import warnings_config # noqa: F401 / RUFF(F401)

import time
#import argparse
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import os
import numpy as np


'''
q_EEG : Queue to send EEG data to main process
q_ICOM_EEG : Queue to receive instructions from main process
q_RCOM_EEG : Queue to send responses back to main process
barrier : Barrier for synchronization with other processes
'''

class EEG_con():
    def __init__(self, serial_port='COM9'):        # '/dev/ttyUSB1' linux or USB_ windows
        # BrainFlow setup
        params = BrainFlowInputParams()
        params.serial_port = serial_port  # BEWARE OF SERIAL_PORT
        board_id = BoardIds.CYTON_DAISY_BOARD
        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()

        print("\nEEG - [OK] EEG collector initialized:")

    def insert_marker(self, board, code):
        board.insert_marker(code)

    def create_file_header(self, filepath):
        # if file doesn't exist, write header

        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists: {filepath}")

        sensor_headers = ['Sample_index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8', 'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13', 'EXG Channel 14', 'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Not Used', 'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)', 'Digital Channel 3 (D17)', 'Not Used', 'Digital Channel 4 (D18)', 'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Marker Channel']

        with open(filepath, 'w', newline='') as f:
            np.savetxt(f, np.array([sensor_headers]),
                    delimiter=',', fmt='%s')

    def start(self, q_log_EEG, q_ICOM_EEG, q_RCOM_EEG, stream_on_event):
        '''
        Calling this method listens for queue instructions and acts accordingly.
        Instructions:
            - Is a tuple of (command, filepath) - Index 0 is command, Index 1 is filepath
            - command: "record" or "stop"
            - filepath: path to save the recorded data (only for "record" command)
        '''

        protocol_never_executed = True

        while True:
            
            if not q_ICOM_EEG.empty():          # Enter if there is a queue
                instruction = q_ICOM_EEG.get()  # Get the instruction from the queue
                
                match instruction[0]:
                    case "record":
                        filepath_EEG = instruction[1]
                        self.create_file_header(filepath = filepath_EEG)
                        
                        exit_while = False
                        self.board.start_stream()
                        while not stream_on_event.is_set():
                            if not exit_while:
                                q_RCOM_EEG.put('True')
                                exit_while = True
                            stream_on_event.wait(timeout=0.01)
                            self.board.get_board_data()     # Flush ring buffer
                        
                        print(f'EEG - Start stream: {time.time()}')
                        protocol_never_executed = False
                        
                    case "stop":
                        if protocol_never_executed:
                            break
                        print('EEG - Stopping EEG recording')
                        data = self.board.get_board_data()
                        self.board.stop_stream()
                        self.board.release_session()

                        with open(filepath_EEG, 'a', newline='') as f:
                            np.savetxt(f, data.T, delimiter=',', fmt='%.6f')
                        break
            
            if not q_log_EEG.empty():
                log_msg = q_log_EEG.get()
                self.board.insert_marker(log_msg)

def test_eeg_streaming():
    params = BrainFlowInputParams()
    params.serial_port = 'COM9'  # BEWARE OF SERIAL_PORT
    board_id = BoardIds.CYTON_DAISY_BOARD
    BoardShim.enable_dev_board_logger()
    board = BoardShim(board_id, params)

    board.prepare_session()
    board.start_stream()
    t0 = time.perf_counter_ns()

    print('sleeping 3 sec')
    board.insert_marker(101)    # start trim
    time.sleep(3)
    print('trim start marker insert', (time.perf_counter_ns() - t0) / 1e9)
    board.insert_marker(102)    # end trim
    print('trim end marker insert', (time.perf_counter_ns() - t0) / 1e9)

    epoch = 50
    for i in range(epoch):
        print('rest')
        board.insert_marker(10)
        time.sleep(1)
        print((time.perf_counter_ns() - t0) / 1e9)

        print('press')
        board.insert_marker(20)
        time.sleep(2)
        print((time.perf_counter_ns() - t0) / 1e9)

        print('rest')
        board.insert_marker(30)
        time.sleep(1)
        print((time.perf_counter_ns() - t0) / 1e9, '\n')

    print('sleeping 3 sec')
    board.insert_marker(201)    # start trim
    time.sleep(3)
    print((time.perf_counter_ns() - t0) / 1e9)
    board.insert_marker(202)    # end trim
    print((time.perf_counter_ns() - t0) / 1e9)

    data = board.get_board_data().T
    board.stop_stream()
    board.release_session()
    np.savetxt('test_EEG_streaming.csv', data, delimiter=',', fmt='%.6f')
    
if __name__ == "__main__":
   test_eeg_streaming()
    