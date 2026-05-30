# SoftPINCH
Real-time EMG decoding for control of a soft robotic hand exoskeleton using deep learning.

## NOTE
This work is part of an ongoing master’s thesis project. Certain implementations, refinements, and code structure optimizations are still under development.

## Overview
This project presents a EMG biosignal framework for real-time control of a soft hand exoskeleton. The framework investigates the individual contributions of EEG and EMG, as well as their decision-level fusion, for decoding hand motor intentions.

Three deep learning architectures are explored:

* LSTM ($N_1$)
* CNN + LSTM ($N_2$)
* CNN + LSTM + Attention ($N_3$)

The system integrates:

* Real-time EMG acquisition
* EEG/EMG preprocessing pipelines
* Neural decoding networks
* Decision-level fusion

## System Architecture

The framework combines:

1) Biosignal acquisition
2) Signal preprocessing
3) Deep neural decoding
4) Motion classification
6) Exoskeleton actuation

## Deep Learning Models
**$N_1$ — LSTM**
Baseline temporal sequence model for biosignal decoding.

**$N_2$ — CNN + LSTM**
Sequential 1D CNN layers extract local temporal and cross-channel features before the LSTM models temporal dependencies.

**$N_3$ — CNN + LSTM + Attention**
Attention mechanism enhances temporal feature weighting and improves discriminative representation learning.

## Repository Structure
The repository contains the following elements:
  - **data_fusion**:
    - data_fusion_manager : This is the key component for data aquistion of both EMG and EEG data given a protocol. EEG can be turned off by chancing the constant "METHOD" : METHOD = '_ _ EMG'.
    - EEG_collector : Used within data_fusion_manager to save EEG data.
    - EMG_collector : Used within data_fusion_manager to save EMG data.
  - **experiment**:
    - experimental_protocol : Used within data_fusion_manager to handle the experimental protocol
    - metabolic_cost_exp : Holds the experimental protocol used in the muscular effort experiments
    - real_time_operation : Used for deployment of traning, real-time inference model
  - **models**:
    - classification_pipeline : Handle every training senario of models.
    - loggings : Contain pretrained models of subject-independent classification and real-time inference systems.
  - **utilities**:
    - Contrains additional functionality used by the other scripts. 

## Environment
Follow these steps to setup the virtual environment:
REQUIREMENT : Python version: 3.11

**Install:**
py -3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

**OLD VERSION CONFLICT** - change these to avoid sherpa shutdown erros:
Follow the path : .venv/lib/python3.11/site-packages/GPyOpt/core/evaluators/batch_local_penalization.py

Do the following change in line 67:
minusL = res.fun[0][0]   ->   minusL = res.fun if isinstance(res.fun, float) else res.fun[0][0]

## Remarks
Trigno implementation only works in windows, at the time being.

data_fusion_manager script with listen_for_terminal_input func only works in windows given command: msvcrt.kbhit()
