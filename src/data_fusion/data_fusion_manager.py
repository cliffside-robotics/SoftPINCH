# Remove warnings regarding API
# from src.utilities import warnings_config # noqa: F401 / RUFF(F401)
from ..utilities import warnings_config # noqa: F401 / RUFF(F401)

# Manage path and arguments
from pathlib import Path
import msvcrt

# Manage structure
from multiprocessing import JoinableQueue, Process, Barrier, Event
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any

# Manage time
from datetime import datetime

# Own implementation
from .EMG_collector import EMG_con as EMG
from .EEG_collector import EEG_con as EEG
#from ..utilities.dynamixel_control import main as MC
from ..utilities.create_record_folders import create_recording_folder
# from ..experiment.experimental_protocol import PROTOCOL_con as PROTOCOL
from ..experiment.metabolic_cost_exp import Exp1BendFingerProtocol as PROTOCOL
       
current_dir = Path(__file__).resolve().parent     # folder of current script
parent_dir = current_dir.parent                   # one level up

#-----------#
# Constants #
#-----------#
METHOD = '_ _ EMG'                                     # Select method in MODES by filled out blank: _ _ _. Where 'all' -> MC EEG EMG
BASE_PATH = str(parent_dir) + r'\experiment\data'       # Where to store DATA
SUBJECT_NAME = "metabolic_cost/isolated_flexion"                 # Name of the subject : subject 0, subject 1
FINGER_NAME = 'flex_thumbActiveExo_finger'                       # Name of the data file      - flex_baseline_finger -> 6 epochs
NUM_EPOCHS = 10                                        # Number of epochs per experiment
REST_DURATION = 1                                       # Rest duration (sec) during 1 trial
ONSET_DURATION = 5                                      # ONSET duration (sec) during 1 trial
REL_DURATION = 5                                        # Release duration (sec) during 1 trial
TRIM_DURATION = 3                                       # Trim duration (sec) in the beginning and end of experiment
MC_PORT = 'COM8'                                        # Define MC port
EEG_PORT = 'COM9'                                       # Define EEG port
EMG_SELECT_SENSORS = (0, 2)                             # EMG data channels. For EMG only: sensor1 = 0, sensor2 = 1, sensor3 = 2
EMG_SAMPLES_PER_READ = 200                             # Samples per read for the EMG sensors
MODES = {
    "MC _ _":    ["MC", "PRO"],
    "MC EEG _":  ["MC", "EEG", "PRO"],
    "MC _ EMG":  ["MC", "EMG", "PRO"],
    "_ EEG _":   ["EEG", "PRO"],
    "_ EEG EMG": ["EEG", "EMG", "PRO"],
    "_ _ EMG":   ["EMG", "PRO"],
    "all":       ["MC", "EEG", "EMG", "PRO"],
}
''' BEFORE EXECUTION
* Note which COM port is in use
* EMG class : select_sensors and samples_per_read can be changed depending on usecase
* SUBJECT_NAME and BASE_PATH can be changed
* listen_for_terminal_input :
    execute_protocol : Change experiment protocol
    file_path : Can add description infront of 'current_time'. Like filepath_MC = folders["MC"] / f"Example_{current_time}.csv"
* Remember to check when processes start and it might need to be shifted
***
Notice protocol is changed to new version
***
''' 

def MC_start(q_MC, q_ICOM_MC, q_RCOM_MC, barrier_init, stream_on_event):
    #mc_ins = MC.Ada_con(co_mod = 0, re_only = False, devicename = MC_PORT)  # linux: '/dev/ttyUSB0', windows: 'COM3'
    barrier_init.wait()
    
    print('MC NOT IMPLEMENTATED WITH STREAM_ON_EVENT and WITHOUT BARRIER_EXEC')
    #mc_ins.start(q_MC, q_ICOM_MC, q_RCOM_MC, stream_on_event)
    #mc_ins.close()

def EEG_start(q_log_EEG, q_ICOM_EEG, q_RCOM_EEG, barrier_init, stream_on_event):
    eeg_ins = EEG(serial_port = EEG_PORT)  # BEWARE OF SERIAL_PORT
    barrier_init.wait()
    
    eeg_ins.start(q_log_EEG, q_ICOM_EEG, q_RCOM_EEG, stream_on_event)
    print('\nEEG - Process is terminated')

def EMG_start(q_log_EMG, q_ICOM_EMG, q_RCOM_EMG, barrier_init, stream_on_event):
    emg_ins = EMG(select_sensors = EMG_SELECT_SENSORS,
                  samples_per_read = EMG_SAMPLES_PER_READ,
                  units = 'mV')
    
    barrier_init.wait()
    
    emg_ins.start(q_log_EMG = q_log_EMG,
                  q_ICOM_EMG = q_ICOM_EMG, 
                  q_RCOM_EMG = q_RCOM_EMG,
                  stream_on_event = stream_on_event)
    print('\nEMG - Process is terminated')

def PROTOCOL_start(q_i_PRO : JoinableQueue,
                   q_r_PRO : JoinableQueue,
                   barrier_init : Any,
                   shutdown_event : Any,
                   stream_on_event : Any,
                   q_log_EEG : JoinableQueue,
                   q_log_EMG : JoinableQueue):
    
    protocol_ins = PROTOCOL(num_epochs = NUM_EPOCHS,
                            rest_duration = REST_DURATION,
                            onset_duration = ONSET_DURATION,
                            release_duration = REL_DURATION,
                            trim_duration = TRIM_DURATION)
    barrier_init.wait()
    
    protocol_ins.start(q_ICOM_PRO = q_i_PRO, 
                       q_RCOM_PRO = q_r_PRO,
                       stream_on_event = stream_on_event,
                       q_log_EEG = q_log_EEG,
                       q_log_EMG = q_log_EMG)
    
    shutdown_event.set()        # Set shutdown_event to terminate all processes
    print('\nPRO - Process is terminated')

#---------------#
# Configuration #
#---------------#
@dataclass
class OWN_PROCESS:
    name : str
    start_func : callable

OWN_PROCESSES = {
    'MC' : OWN_PROCESS('MC', MC_start),
    'EEG' : OWN_PROCESS('EEG', EEG_start),
    'EMG' : OWN_PROCESS('EMG', EMG_start),
    'PRO' : OWN_PROCESS('PRO', PROTOCOL_start),
}

def send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instruction, method):
    active = MODES[method]
    mapping = {'MC' : q_i_MC,
               'EEG' : q_i_EEG,
               'EMG' : q_i_EMG,
               'PRO' : q_i_PRO
               }
    
    for i, key in enumerate(['MC', 'EEG', 'EMG', 'PRO']):
        if key in active:
            mapping[key].put(instruction[i])

def listen_for_terminal_input(q_i_MC : Optional[JoinableQueue],
                              q_i_EEG : Optional[JoinableQueue],
                              q_i_EMG : Optional[JoinableQueue],
                              q_i_PRO : Optional[JoinableQueue], 
                              barrier_init : any,
                              select_method : Dict,
                              shutdown_event : any):
    """Listen for terminal input and send commands to the queue."""
    barrier_init.wait()
    command = None
    
    print('\nWrite "record" to start protocol and write "stop" to end execution')
    while not shutdown_event.is_set():
        
        if msvcrt.kbhit():                     # key pressed?
            command = input("\nEnter command: ").strip().lower()
        
        if command == "record":
            
            folders = create_recording_folder(SUBJECT_NAME, BASE_PATH)

            current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

            instructions = []
            for key in ['MC', 'EEG', 'EMG', 'Markers']:
                filepath = str(folders[key]) + f"/{FINGER_NAME}_{current_time}.csv" 
                instruction_temp = (command, filepath)
                instructions.append(instruction_temp)
            
            send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)
            command = None      # Reset back to None

        elif command == "stop":
            instructions = [('stop', None)] * 4
            send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)
            break

        elif command is not None:
            print("Unknown command. Please enter 'record' or 'stop'.")
            command = None
    
    if shutdown_event.is_set():
        instructions = [('stop', None)] * 4
        send_command_queue(q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, instructions, select_method)

def check_sensor_status(q_r_EEG : JoinableQueue,
                        q_r_EMG : JoinableQueue,
                        stream_on_event : Any,
                        sensor_usage : str):
    '''
    The protocol process sends messages to the EEG and EMG queues when they are ready to start streaming.
    This function listens to those queues and sets the stream_on_event when both sensors are ready.
    '''
    if sensor_usage == 'EMG_only':                      # Only consider EMG sensor to be active
        EEG_ready, EMG_ready = True, False
    else:                                               # Consider both sensors to be active
        EEG_ready, EMG_ready = False, False

    msg = 'False'

    while not stream_on_event.is_set():

        if not EEG_ready and not q_r_EEG.empty():
            msg = q_r_EEG.get()

            if msg == 'True':
                EEG_ready = True
                msg = 'False'
                print('EEG_ready = True')
        
        if not EMG_ready and not q_r_EMG.empty():
            msg = q_r_EMG.get()

            if msg == 'True':
                EMG_ready = True
                msg = 'False'
                print('EMG_ready = True')
        
        if EEG_ready and EMG_ready:
            stream_on_event.set()
    
    print('Exit - check_sensor_status')

def build_system(active_modes : Dict):
    """
    Build queues, processes, and barriers corresponding to the number of active processes.

    Parameters
    ----------
    active_modes : Dict
        Dictionary describing the selected MODE and its active processes.

    Returns
    -------
    queues : Dict
        Mapping from sensor name to a tuple of communication queues:
        (q_main, q_ICOM, q_RCOM).
        Example:
            {
                "MC":  (JoinableQueue, JoinableQueue, JoinableQueue),
                "EEG": (JoinableQueue, JoinableQueue, JoinableQueue),
                ...
            }

    processes : list[multiprocessing.Process]
        List of all spawned sensor processes.

    barrier_init : multiprocessing.Barrier
        Barrier used to synchronize initialization of all sensor processes.

    shutdown_event : multiprocessing.Event
        Whenever the experimental protocol terminates. shutdown_event is set and allows all processes to terminate via listen_for_terminal_input()
    
    stream_on_event : multiprocessing.Event
        Event Trigger when EEG and EMG is started and streaming data. Ensure syncronization between processes
    """
    queues = {}
    processes = []

    num_modes = len(active_modes)
    barrier_init = Barrier(num_modes + 1)     # Purpose: To hold processes until all is initilized + listen_for_terminal_input Thread
    stream_on_event = Event()                 # Event trigger. Syncronizes EEG, EMG and protocol. When data stream is ready

    for key in active_modes:
        # Queues for inter-process communications
        q_log = JoinableQueue(100)                 # Purpose: Queue to main process
        q_i = JoinableQueue(100)                    # Purpose: Queue to receive instructions from main process
        q_r = JoinableQueue(100)                    # Purpose: Queue to send responses back to main process

        queues[key] = (q_log, q_i, q_r)            # Load queues into dict with key-ID

        if key == 'PRO':
            shutdown_event = Event()                # Purpose: Whenever protocol terminates, set this true and it will terminate all processes
            q_log_EEG = queues.get('EEG', (None, None, None))[0]
            q_log_EMG = queues.get('EMG', (None, None, None))[0]
            args = (q_i, q_r, barrier_init, shutdown_event, stream_on_event, q_log_EEG, q_log_EMG)        # Pass q_i_EEG and q_i_EMG to protocol process for sending 'marker' messages
            
        else:
            args = (q_log, q_i, q_r, barrier_init, stream_on_event)

        process_temp = Process(
            target = OWN_PROCESSES[key].start_func,
            args = args
        )
        processes.append(process_temp)
    
    return queues, processes, barrier_init, shutdown_event, stream_on_event

def main():
    if METHOD not in MODES:
        raise ValueError('Invalid method')

    active = MODES[METHOD]             # Extract the mode from the desired argument

    queues, processes, barrier_init, shutdown_event, stream_on_event = build_system(active)

    # What is [1] -> Get the q_i for each process.
    # If a process is not active, default set value (q_log, q_i, q_r) to None 
    q_i_MC = queues.get('MC', (None, None, None))[1]
    q_i_EEG = queues.get('EEG', (None, None, None))[1]
    q_i_EMG = queues.get('EMG', (None, None, None))[1]
    q_i_PRO = queues.get('PRO', (None, None, None))[1]
    
    q_r_EEG = queues.get('EEG', (None, None, None))[2]
    q_r_EMG = queues.get('EMG', (None, None, None))[2]
    
    # Thread for user input
    terminal = threading.Thread(
        target = listen_for_terminal_input,
        args = (q_i_MC, q_i_EEG, q_i_EMG, q_i_PRO, barrier_init, METHOD, shutdown_event),
        daemon = True
    )

    # Quick fix:
    #   Allow check_sensor_status will only consider EMG responds.
    #   Ignore EEG sensor
    if METHOD == '_ _ EMG':                 
        sensor_usage = 'EMG_only'
    else:
        sensor_usage = 'EMG_EEG'

    # Respond from processes using sensors
    start_streaming = threading.Thread(
        target = check_sensor_status,
        args = (q_r_EEG, q_r_EMG, stream_on_event, sensor_usage),
        daemon = True
    )

    for p in processes:
        p.start()

    terminal.start()
    start_streaming.start()

if __name__ == '__main__':
    main()
