# Computational-Modeling-of-a-Fluid-Catalytic-Cracking-Unit

This repository contains the code used to reproduce the multi-step, multivariate forecasting for Fluid Catalytic Cracking (FCC) product yields presented in the paper:

> **Computational modeling of a Fluid Catalytic Cracking Unit**  
> Mustapha K. Khaldi *et al.*

The implementation covers several neural-network architectures:
- **Dense** and **CNN** baselines
- Various **LSTM** variants (vanilla, Stacked, Bidirectional, Encoder–Decoder, CNN–LSTM)
- **Multi-Headed LSTM (MH-LSTM)** with separate heads for reactor, regenerator and fractionator subsystems

## Repository Structure

```
├── Multistep_model_train.ipynb    # Jupyter notebook demonstrating training/evaluation
├── NN_net.py                      # Model definitions (Dense, CNN, LSTM variants) fileciteturn0file1
├── utils.py                       # Utility functions for data scaling, plotting, metrics fileciteturn0file2
├── WindowGen.py                   # WindowGenerator for sliding-window dataset creation fileciteturn0file3
├── requirements.txt               # Core Keras/TensorFlow dependencies
└── README.md                      # (this file)
```

## Installation

1. **Clone** this repository:  
   ```bash
   git clone https://github.com/your-username/Computational-Modeling-of-a-Fluid-Catalytic-Cracking-Unit.git
   cd Computational-Modeling-of-a-Fluid-Catalytic-Cracking-Unit
   ```

2. **Create** and **activate** a virtual environment (e.g. `venv` or `conda`).

3. **Install** the core Keras/TensorFlow dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   where `requirements.txt` contains:
   ```
   keras==2.10.0
   Keras-Preprocessing==1.1.2
   keras-tuner==1.4.7
   tensorflow==2.10.1
   tensorflow-estimator==2.10.0
   ```
   fileciteturn0file4

> *Note:* For the full list of packages used in development, see `req.txt`.

## Usage

- **Notebook**: Open and run `Multistep_model_train.ipynb` to step through data preparation, model training, evaluation, and plotting.
- **Scripts**:  
  - Define or modify hyperparameters in the notebook or import functions from `NN_net.py`.  
  - Use `utils.evaluate_forecast` and `utils.plot_forecast` to generate performance metrics and figures.

## Citation

If you use this code in your work, please cite:

> Khaldi, M. K., Al-Dhaifallah, M., Taha, O., Mahmood, T., & Alharbi, A. (2025). Computational modeling of a Fluid Catalytic Cracking Unit. Ain Shams Engineering Journal 2025

---

Feel free to open issues or submit pull requests for enhancements!
