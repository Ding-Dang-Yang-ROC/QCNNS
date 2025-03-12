# Simulating Nonlinearity in Quantum Neural Networks While Mitigating Barren Plateaus

## Author
**Ding-Dang Yang**  
Department of Economics, National Chengchi University, Taipei, 116, Taiwan, R.O.C.  
E-mail: 112258018@g.nccu.edu.tw  

## Installation
To install dependencies, use the following commands:
```sh
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge pennylane=0.38.0
```

## Code Description

### Directories and Files
```
1_mnist/                # Implementation for the MNIST dataset.
2_fmnist/               # Implementation for the FMNIST dataset.
1_mnist/1_data/         # Downloads the MNIST dataset and converts it to (8×8) pixel format.
1_mnist/1_data/mnist/8_8  # MNIST dataset with (8×8) pixel format used for training and testing.
2_08_08/1_classical/    # Direct Pixel-to-CFC Mapping, corresponding to the results in Table 2.
1_6_qubits/             # Uses 2^6 (64) pixels.
2_12_qubits/            # Uses tensor product of 2^6 (64) and 2^6 (64) pixels.
2_08_08/2_quantum/      # Quantum method, corresponding to the results in Table 1.
  ├── 1_con4_1_L/       # QCNN with 1 QFilter layer.
  ├── 2_con4_2_L/       # QCNN with 2 QFilter layers.
  ├── 3_con4_3_L/       # QCNN with 3 QFilter layers.
Result/                 # Stores results from 1_classical and 2_quantum.
  ├── accu_case_ori.gz  # Compares predicted and target values for training and testing datasets at every 10 epochs.
  ├── accu_sum          # Accuracy of training and testing datasets at every 10 epochs.
  ├── par               # Records the model parameters after training.
3_noise/                # Introduces quantum noise using depolarizing and phase-damping channels.
  ├── 1_default.qubit/  # Reads trained parameters and calculates accuracy.
  ├── 2_default.mixed/  # Reads trained parameters, adds noise, and calculates accuracy.
```

## Usage
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd <repo_name>
   ```
2. Set up the environment using Conda (see Installation section above).
3. Run experiments using provided scripts in `1_mnist/`, `2_fmnist/`, and `3_noise/`.
