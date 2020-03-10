import functools
from typing import Union, Tuple, Any

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
from tqdm import tqdm


draw = plt.plot


def exponential_smoothing(series, alpha):
    result = np.array([series[0]])  # first value is same as series
    for n in range(1, len(series)):
        result = np.append(result, alpha * series[n] + (1 - alpha) * result[n - 1])

    res = 0
    for s, r in zip(series, result):
        res += (s - r) ** 2
    return result


def find_aic(array1: Union[list, np.ndarray], array2: Union[list, np.ndarray], k: int = 5):
    """
        Поиск КФ Акаике
    :param array1: Первый список
    :param array2: Второй список
    :param k: Количество параметров
    :return: КФ Акаике
    """
    a = array1.copy()
    b = array2.copy()
    if not isinstance(a, list):
        a = a.tolist()
    if not isinstance(b, list):
        b = b.tolist()
    return rp.aic.aic(a, b, k) / len(a)


def find_bic(array1: Union[list, np.ndarray], array2: Union[list, np.ndarray], k: int = 5):
    """
        Вычисление критерия Шварца
    :param array1: Первый список
    :param array2: Второй список
    :param k: Количество параметров
    :return: Критерий Шварца
    """
    a = array1.copy()
    b = array2.copy()
    if not isinstance(a, list):
        a = a.tolist()
    if not isinstance(b, list):
        b = b.tolist()
    return rp.bic.bic(a, b, k) / len(a)


def get_polyfit(seq, k: int) -> Tuple[np.ndarray, Any, np.ndarray]:
    p = polyfit(range(len(seq)), seq, k)
    return p


def find_trend_by_kf(seq, p: Union[np.ndarray, int, tuple]):
    if isinstance(p, tuple):
        p = p[0]
    elif isinstance(p, int):
        p = get_polyfit(seq, p)
    pf = np.poly1d(p)
    return pf(range(len(seq)))


def find_determination_kf(seq, k):
    pl = find_trend_by_kf(seq, get_polyfit(seq, k))
    mean = seq.mean()
    top = ((pl-mean)**2).sum()
    bottom = ((seq-mean)**2).sum()
    determination = top/bottom
    return determination


def find_adjusted_determination_kf(seq, k):
    determination_kf = find_determination_kf(seq, k)
    # XXX -5
    kf = 1 - (1 - determination_kf) * ((len(seq) - 1) / (len(seq) - 5))
    return kf


def find_acf(seq, a, b=None):
    """
        Поиск автокорреляции
    :param seq: Последовательность
    :param a: Поиск ОТ
    :param b: Поиск ДО
    :return: Список/Значение корреляции
    """
    lags = []
    if b is None:
        return pearsonr(seq[:-a], seq[a:])[0]
    for lag in range(a, b+1):
        res_lag = pearsonr(seq[: -lag], seq[lag:])[0]
        lags.append(res_lag)
    return lags


class Arima:
    def __init__(self, endog, *args, **kwargs):
        self.model = ARIMA(endog, *args, **kwargs)
        if kwargs.get('order') is not None:
            self.model.order = kwargs['order']

    def fit(self, *args, **kwargs):
        _fitted = self.model.fit(*args, **kwargs)
        _fitted.data = _fitted.fittedvalues.copy()
        _fitted.old_aic = _fitted.aic
        _fitted.aic = find_aic(self.model.endog, _fitted.data)
        if hasattr(self.model, 'order'):
            _fitted.order = self.model.order
        return _fitted

    @classmethod
    def find_optimal_model_by_order(cls, endog, p_sequence, d_sequence, q_sequence):
        if isinstance(p_sequence, int):
            p_sequence = (p_sequence, )
        if isinstance(d_sequence, int):
            d_sequence = (d_sequence, )
        if isinstance(q_sequence, int):
            q_sequence = (q_sequence, )
        min_fitted = None

        iter_count: int = len(p_sequence) * len(d_sequence) * len(q_sequence)
        for p, d, q in tqdm(
                itertools.product(p_sequence, d_sequence, q_sequence),
                total=iter_count, desc='ARIMA ', unit=' fit'):
            try:
                _model = cls(endog, order=(p, d, q))
                _fitted = _model.fit()
                if min_fitted is None or min_fitted.aic > _fitted.aic:
                    min_fitted = _fitted
            except:
                pass
        return min_fitted.model, min_fitted

    def __getattr__(self, item):
        return getattr(self.model, item)


if __name__ == '__main__':
    series = [i for i in pd.read_csv('./data/RTS-history-USA.csv')[::25].Open]
    model, fitted = Arima.find_optimal_model_by_order(series, range(5), (0, 1), range(3))
    print(fitted.aic)
