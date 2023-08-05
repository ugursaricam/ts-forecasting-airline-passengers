import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/airline-passengers.csv", parse_dates=True)

df.head()

df.info()

df.isnull().sum()

df["month"] = pd.to_datetime(df["month"])

df = df.set_index("month").squeeze()

def ts_decompose(dataframe, model="additive", stationary=False):
    result = seasonal_decompose(dataframe, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(dataframe, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

ts_decompose(df)

df.shape # (144,)

train = df[:115]
test = df[115:]

############################################
# Triple Exponential Smoothing (Holt-Winters)
############################################

alphas = betas = gammas = np.arange(0.20, 1, 0.10)
abg = list(itertools.product(alphas, betas, gammas))

test.shape # (29,)

# Optimized for best mae
def tes_optimizer(train, abg, step=29):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)
#best_alpha: 0.3 best_beta: 0.5 best_gamma: 0.9 best_mae: 13.3118

tes_model = ExponentialSmoothing(train,
                                       trend="add",
                                       seasonal="add",
                                       seasonal_periods=12).fit(smoothing_level=best_alpha,
                                                                smoothing_trend=best_beta,
                                                                smoothing_seasonal=best_gamma)

y_pred_tes_model = tes_model.forecast(29)

def plot_prediction(y_pred, label):
    train.plot(legend=True, label="TRAIN")
    test.plot(legend=True, label="TEST")
    y_pred.plot(legend=True, label="PREDICTION")
    plt.title("Train, Test and Predicted Test Using "+label)
    plt.show()

plot_prediction(y_pred_tes_model, "Triple Exponential Smoothing ADD")

############################################
# SARIMA (Seasonal ARIMA)
############################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# for best aic
def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

best_order #(1, 1, 0)
best_seasonal_order #(0, 1, 0, 12)

sarima_model_aic = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order).fit(disp=0)
y_pred_sarima_model_aic = sarima_model_aic.get_forecast(steps=29)

y_pred_sarima_model_aic = y_pred_sarima_model_aic.predicted_mean

mean_absolute_error(y_pred_sarima_model_aic,test) # 28.547408823346995


def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show(block=True)

plot_co2(train, test, y_pred_sarima_model_aic, "SARIMA")


# for best mae
def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), None, None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=29)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

best_order # (0, 0, 0)
best_seasonal_order # (1, 1, 1, 12)

sarima_model_mae = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order).fit(disp=0)
y_pred_sarima_model_mae = sarima_model_mae.get_forecast(steps=29)

y_pred_sarima_model_mae = y_pred_sarima_model_mae.predicted_mean

mean_absolute_error(y_pred_sarima_model_mae,test) # 20.222706191173742

plot_co2(train, test, y_pred_sarima_model_mae, "SARIMA")


plt.figure(figsize=(15,6))
test.plot(label="TEST")
y_pred_tes_model.plot(label="TES MODEL")
y_pred_sarima_model_aic.plot(label="SARIMA MODEL AIC")
y_pred_sarima_model_mae.plot(label="SARIMA MODEL MAE")
plt.legend()


all_model = pd.DataFrame()
all_model["test"] = test
all_model["y_pred_tes_model"] = y_pred_tes_model
all_model["y_pred_sarima_model_aic"] = y_pred_sarima_model_aic
all_model["y_pred_sarima_model_mae"] = y_pred_sarima_model_mae

all_model.head()
#             y_pred_tes_model  y_pred_sarima_model_aic  y_pred_sarima_model_mae  test
# 1958-08-01        472.959663               490.337911               510.156710   505
# 1958-09-01        408.820127               427.883043               439.868490   404
# 1958-10-01        354.278868               370.771413               376.917276   359
# 1958-11-01        316.674292               328.794272               331.362149   310
# 1958-12-01        352.856388               359.789591               363.539988   337