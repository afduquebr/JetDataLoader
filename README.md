
---

# JetDataLoader

**Author:** Andrés Felipe Duque Bran

***

## Overview

JetDataLoader is a modular framework for preprocessing, training, and evaluating deep learning models for jet classification in particle physics. It supports Particle Flow Networks (PFN), Particle Transformer (ParT), and ParticleNet architectures, providing flexibility for handling different models and training strategies. The repository is designed for clean workflows, from data preparation to final model evaluation and plotting.

> ⚠️ Note: The framework only supports **HDF5 (.h5)** files for input data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Repository Structure](#repository-structure)

## Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/afduquebr/JetDataLoader.git
   cd JetDataLoader
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. **Preprocessing**:  
   Use the executable script inside the `exec` directory to preprocess raw data into training-ready format.  
   The preprocessing pipeline includes:
   - Skimming
   - Reweighting
   - Class balancing and shuffling. 
   
   Example:
   ```bash
   ./exec/preprocessing.sh
   ```

   The `skim.json` file controls variables kept, metadata employed, and other preprocessing details.

### Model Training and Evaluation

1. **Training**:  
   Use the training script to train PFN, ParT, or ParticleNet models. 

   Example:
   ```bash
   ./exec/train.sh
   ```

2. **Evaluation and Plotting**:
   After training, evaluate the model and generate plots with:

   ```bash
   ./exec/eval.sh
   ```

   The output includes:
   - Model performance plots
   - Input feature distributions
   - Prediction distributions

## Repository Structure

```
JetDataLoader
│
├── config/                  # YAML config files for variables and metadata
├── exec/                    # Executable scripts (preprocessing, training)
├── figs/                    # Figures of input features and model outputs
├── logs/                    # Logs from training and preprocessing
├── models/                  # Saved model weights (.pt files)
├── networks/                # Model architecture definitions (PFN, ParT, ParticleNet)
├── plot/                    # Plotting scripts for inputs and outputs
├── utils/                   # Utility scripts (preprocessing, skimming, balancing, etc.)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Notes

- The `config/config.yaml` file centralizes experiment settings, including data paths, model hyperparameters, and training options.
- Logs and model checkpoints are automatically saved for reproducibility.
- Feel free to expand the `networks/` directory to include additional architectures as needed!

***