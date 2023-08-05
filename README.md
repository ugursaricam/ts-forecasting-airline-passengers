# Airline Passengers Forecasting with SARIMA

![Dataset](https://img.shields.io/badge/dataset-airline_passengers.csv-blue.svg) ![Python](https://img.shields.io/badge/python-3.9-blue.svg) ![SARIMA](https://img.shields.io/badge/SARIMA-hyperparam_optimization-blue.svg)

This GitHub repository contains the implementation of a SARIMA (Seasonal Autoregressive Integrated Moving Average) model for forecasting airline passenger numbers using the "airline-passengers.csv" dataset. The dataset consists of two columns: "month" and "total_passengers." The SARIMA model has been constructed by optimizing hyperparameters to accurately predict passenger numbers.

## Dataset

The dataset "airline-passengers.csv" contains monthly data of total airline passengers. It includes the following columns:

- `month`: The month of the observation.
- `total_passengers`: The total number of passengers in the respective month.

## Installation

To run the code locally, follow these steps:

1. Clone the repository:
git clone https://github.com/ugursaricam/ts-forecasting-airline-passengers.git

2. Install the required Python packages:
pip install -r requirements.txt

3. Explore the Jupyter Notebook or Python files for the SARIMA model and hyperparameter optimization.

## Model Construction

In this project, a Seasonal Autoregressive Integrated Moving Average (SARIMA) model has been utilized for time series forecasting. SARIMA is a popular method for handling seasonality in time series data and can be useful for predicting the total number of airline passengers over time. The hyperparameters of the SARIMA model have been optimized to improve its forecasting accuracy.

## Contributions and Feedback

If you would like to contribute to this project or provide feedback, feel free to open a new issue or submit a pull request. Your input is valuable for enhancing the accuracy and performance of the forecasting model. Let's collaborate to create a more powerful forecasting solution!

---
Please note that the information provided above is just an example. Feel free to customize it with relevant details about your specific project and dataset. The goal is to provide a clear overview of the repository and its contents to potential users and contributors. 

Happy coding!
