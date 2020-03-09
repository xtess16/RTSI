import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
import numpy as np
from numpy import polyfit
from scipy import stats
import RegscorePy as rp
from numpy import sqrt
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
import itertools
import chow_test


def find_aic(array1, array2, k=5):
    a = array1.copy()
    b = array2.copy()
    if not isinstance(a, list):
        a = a.tolist()
    if not isinstance(b, list):
        b = b.tolist()
    return rp.aic.aic(a, b, k) / len(a)


class Arima(ARIMA):
    def __new__(cls, endog, *args, **kwargs):
        print('a')
        # if not isinstance(endog, np.ndarray):
        #     self._base_model = np.array(endog)
        # else:
        #     self._base_model = endog
        return super().__new__(cls, endog, *args, **kwargs)

    def __init__(self, endog, *args, **kwargs):
        super().__init__(endog, *args, **kwargs)

    def fit(self, *args, **kwargs):
        print('f')
        fitted = super().fit()
        print('f1')
        fitted.data = self.endog - fitted.resid
        fitted.aic2 = find_aic(self.endog, fitted.data)
        print(fitted.aic2)
        return fitted
