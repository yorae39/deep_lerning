"FORMULAS"
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt



y_true = [1, 0, 1, 0]
y_pred = [0.300, 0.020, 0.890, 0.320]
mean_squared_error(y_true, y_pred)

mean_absolute_error(y_true, y_pred)
rms = sqrt(mean_squared_error(y_true, y_pred))
print(rms)


def rmse(predictions, targets):
    return sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(np.array(y_true), np.array(y_pred))
print("rms error is: " + str(rmse_val))

def difValues(a, b):
    return a - b

resultado = difValues(0, 0.320)
# 0,7 / -0,02 / 0,11 / -0,32

