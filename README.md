# Forecasting Under Uncertainty: Application of Temporal Fusion Transformers to Energy Finance

In this repository, we conduct some experiments for real-world use cases using the TFT architecture of Lim et al. (2019).


Each sandbox folder should contain:

- Short description of the use case: description.txt
- Trained/model: lightning_logs and/or model.zip (only locally, not on GitHub)
- Set of settings and hyperparameters: params.py
- Run script to run/train the model: run_model.py
- Some notebook to analyze results: results.ipynb 


## Empirical study Case 1:

- Forecasting Day-Ahead prices (based on generation, demand and weather)

Our data consists of hourly Day ahead prices for Germany/Luxembourg that were saved on a hourly bases from 01/01/2019 until 01/05/2025.
Then, we will try different architectures for the TFT as well as different sets of parameters and hyper-parameters.

