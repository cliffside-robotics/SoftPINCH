from .pytrigno import MVC
from .create_record_folders import create_recording_folder
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

SUBJECT_NAME = "subject_1"                                   # Define the subject name                  
current_dir = Path(__file__).resolve().parent              # folder of current script
parent_dir = current_dir.parent
BASE_PATH = str(parent_dir) + r'\experiment\data\metabolic_cost\isolated_flexion'    # Define the base path for data storage

channel_names =[
'ch1 : Flexor digitorum superficialis',
'ch2 : Flexor pollicis longus'
'ch3 : ...'
]

# channel_names =[
# 'ch1 : Flexor digitorum superficialis',
# 'ch2 : Flexor pollicis longus', 
# 'ch3 : Extensor digitorum',
# 'ch4 : Extensor pollicis brevis/longus'
# ]

''' How to calibrate new data:
normalized_data = (raw_data - baseline_noise) / (MVC - baseline_noise) 
'''


def main():
    # folders = create_recording_folder(SUBJECT_NAME=SUBJECT_NAME, BASE_PATH=BASE_PATH)
    # calibration_path = folders['BASE'] / 'calibration_stats.json'

    # if calibration_path.exists():
        # raise FileExistsError(f'Calibration file already exists : {calibration_path}')

    mvc = MVC(channel_range = (0, 2), samples_per_read = 200, units = 'mV')

    baseline_noise, baseline_mvc = mvc.start_mvc_protocol(rest_window_sec = 6,      
                                                 contract_window_sec = 6,
                                                 repetition = 10)
        
    columns = [f'ch_{i}' for i in range(baseline_noise.shape[1])]
    df_noise = pd.DataFrame(baseline_noise, columns=columns)
    df_mvc = pd.DataFrame(baseline_mvc, columns=columns)

    df_noise.to_csv(fr'{BASE_PATH}\MVC_baseline_noise.csv', index=False)
    df_mvc.to_csv(fr'{BASE_PATH}\MVC_baseline_mvc.csv', index=False)
    
    print('[Success] - calibration performed and written to file')

if __name__ == '__main__':
    main()
    print(BASE_PATH)
    data0 = pd.read_csv(f'{BASE_PATH}/MVC_baseline_noise.csv')
    data1 = pd.read_csv(f'{BASE_PATH}/MVC_baseline_mvc.csv')
    plt.figure()
    plt.plot(data0)
    plt.figure()
    plt.plot(data1)
    plt.show()



