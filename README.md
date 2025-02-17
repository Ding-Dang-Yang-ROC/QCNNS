# Simulating Nonlinearity in Quantum Neural Networks with Reduced Barren Plateaus

## Author
**Ding-Dang Yang**  
Department of Economics, National Chengchi University, Taipei, 116, Taiwan, R.O.C.  
E-mail: 112258018@g.nccu.edu.tw  

## Installation
```sh
conda install numpy pytorch -c pytorch
pip install pennylane
```

## Code Description

### Directories and Files
- `1_mnist/`: Implementation for the MNIST dataset.
- `2_fmnist/`: Implementation for the FMNIST dataset.
- `1_mnist/1_data/`: Downloads the MNIST dataset and converts it to (8×8) pixel format.
- `2_08_08/1_classical/`: Classical method, corresponding to the results in Table 2.
- `2_08_08/2_quantum/`: Quantum method, corresponding to the results in Table 1.

### Notes
- `.gz` files are compressed archives. Use `gunzip` to decompress them.
