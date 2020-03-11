import functools
import traceback
from typing import Union, Sequence, Optional, Tuple, List

import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
import numpy as np
import RegscorePy as rp
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
import itertools
import chow_test
from tqdm import tqdm


T_SEQUENCE = Union[Sequence, np.ndarray]


def draw(seq, *args, **kwargs):
    if isinstance(seq, Model):
        seq = seq.sequence
    if isinstance(seq, PartialModel):
        for s in seq.sequence_for_draw():
            draw(s)
        return
    return plt.plot(seq, *args, **kwargs)


class Model:
    def __init__(self, seq: T_SEQUENCE, k=None):
        self.sequence: np.ndarray = np.array(seq)
        self.k = k

    @property
    def seq(self):
        return self.sequence

    @seq.setter
    def seq(self, value: T_SEQUENCE):
        self.sequence = value

    def __getattr__(self, item):
        return getattr(self.sequence, item)

    def __getitem__(self, item):
        return Model(self.sequence[item])

    def __iter__(self):
        return iter(self.sequence)

    def __sub__(self, other):
        if isinstance(other, Model):
            return Model(self.sequence - other.sequence)
        return Model(self.sequence - other)

    def __add__(self, other):
        if isinstance(other, Model):
            return Model(self.sequence + other.sequence)
        return Model(self.sequence + other)

    def __mul__(self, other):
        if isinstance(other, Model):
            return Model(self.sequence * other.sequence)
        return Model(self.sequence * other)

    def __pow__(self, power):
        return Model(self.sequence ** power)

    def __div__(self, other):
        if isinstance(other, Model):
            return Model(self.sequence / other.sequence)
        return Model(self.sequence / other)

    def __floordiv__(self, other):
        if isinstance(other, Model):
            return Model(self.sequence // other.sequence)
        return Model(self.sequence // other)

    def __len__(self):
        return len(self.sequence)

    def append(self, other, inplace=False):
        """
            Склеивает последовательности
        :param other: Последовательность, которая будет приклеена
        :param inplace: Если True, то изменит список текущей модели, False - вернет новый
        :return: Склеенный список
        """
        if inplace:
            if isinstance(other, Model):
                self.sequence = np.append(self.sequence, other.sequence)
            else:
                self.sequence = np.append(self.sequence, other)
        else:
            if isinstance(other, Model):
                return Model(np.append(self.sequence, other.sequence))
            return Model(np.append(self.sequence, other))

    def fuller_test(self):
        """ Тест Фуллера """
        return adfuller(self.sequence)

    def durbin_watson_test(self):
        """ Тест Дарбина Уотсона """
        dw = 0
        for t in range(1, len(self.sequence)):
            dw += (self.sequence[t] - self.sequence[t-1]) ** 2
        dw /= (self.sequence ** 2).sum()
        return dw

    def arima(self, *args, **kwargs):
        return Arima(self.sequence, *args, **kwargs)

    def acf(self, lag: int):
        """
            Поиск автокорреляции
        :param lag: Значение лага
        :return: Список/Значение корреляции
        """
        return pearsonr(self.sequence[:-lag], self.sequence[lag:])[0]

    def many_acf(self, lag_from: int, lag_to: int):
        """
            Поиск автокорреляции по нескольким лагам
        :param lag_from: Поиск ОТ лага
        :param lag_to: Поиск ДО лага
        :return: Список/Значение корреляции
        """
        lags = []
        for lag in range(lag_from, lag_to + 1):
            res_lag = pearsonr(self.sequence[: -lag], self.sequence[lag:])[0]
            lags.append(res_lag)
        return lags

    def aic(self, array: T_SEQUENCE, k: Optional[int] = None) -> float:
        """
            Поиск КФ Акаике
        :param array: Второй список
        :param k: Количество параметров
        :return: КФ Акаике
        """
        if isinstance(array, Model):
            if k is None:
                k = array.k
            array = array.sequence.tolist()
        elif isinstance(array, np.ndarray):
            array = array.tolist()
        return rp.aic.aic(self.sequence.tolist(), array, k) / len(self.sequence)

    def bic(self, array: T_SEQUENCE, k: Optional[int] = None):
        """
            Вычисление критерия Шварца
        :param array: Второй список
        :param k: Количество параметров
        :return: Критерий Шварца
        """
        if isinstance(array, Model):
            if k is None:
                k = array.k
            array = array.sequence.tolist()
        elif isinstance(array, np.ndarray):
            array = array.tolist()
        return rp.bic.bic(self.sequence.tolist(), array, k) / len(self.sequence)

    def trend(self, p):
        if isinstance(p, int):
            p = self.polyfit(p)
        pf = np.poly1d(p)
        return Model(pf(range(len(self.sequence))), k=len(p))

    def polyfit(self, k: Optional[int] = None):
        if k is None:
            k = self.k
        p = np.polyfit(range(len(self.sequence)), self.sequence, k)
        return p

    def determination_kf(self, k: Optional[int] = None) -> float:
        """
            Подсчет КФ детерминации
        :param k: Количество параметров модели
        :return: КФ детерминации
        """
        if k is None:
            k = self.k
        pl = self.trend(self.polyfit(k)).sequence
        mean = self.sequence.mean()
        top = ((pl - mean) ** 2).sum()
        bottom = ((self.sequence - mean) ** 2).sum()
        determination = top / bottom
        return determination

    def partial_determination_kf(self, other) -> float:
        """
            Подсчет КФ детерминации
        :param other: Кусочный тренд
        :return: КФ детерминации
        """
        pl = other.sequence
        mean = self.sequence.mean()
        top = ((pl - mean) ** 2).sum()
        bottom = ((self.sequence - mean) ** 2).sum()
        determination = top / bottom
        return determination

    def fisher(self, k: Optional[int] = None):
        """
            Критерий Фишера
        :param k: Количество параметров модели
        :return:
        """
        if k is None:
            k = self.k
        return (self.determination_kf(k) / (k-1)) / ((1 - self.determination_kf(k)) / (len(self.sequence) - k))

    def adjusted_determination_kf(self, k: Optional[int] = None):
        """
            Подсчет скорректированного КФ детерминации
        :param k: Количество параметров модели
        :return: Скорректированный КФ детерминации
        """
        if k is None:
            k = self.k
        determination_kf = self.determination_kf(k)
        kf = 1 - (1 - determination_kf) * ((len(self.sequence) - 1) / (len(self.sequence) - k))
        return kf

    def exponential_smoothing(self, alpha):
        """
            Экспоненциальное сглаживание
        :param alpha: КФ
        :return:
        """
        # TODO: найти либу
        result = np.array([self.sequence[0]])
        for n in range(1, len(self.sequence)):
            result = np.append(result, alpha * self.sequence[n] + (1 - alpha) * result[n - 1])
        res = 0
        for s, r in zip(self.sequence, result):
            res += (s - r) ** 2
        return Model(result)

    def cut_by_chow_test(self, *args, arbitrarily: Optional[int] = None, with_result=False) -> \
            Union[Tuple[float, 'PartialModel'], 'PartialModel']:
        """
            Поиск оптимального разделения графика по Чоу тесту
        :param args: Набор диапазонов
        :param arbitrarily: Если передано число,
            то метод сам разделит последовательность на arbitrarily частей
        :param with_result: Возвращать результат теста или нет
        :return: Максимальный результат теста и индексы разделения
        """
        # Если передано arbitrarily, то перебираем все значения
        if isinstance(arbitrarily, int):
            args = tuple(range(len(self.sequence)) for _ in range(arbitrarily-1))
        max_chow_result: float = 0
        max_chow_result_indexes: Optional[List[int]] = None
        iter_count = functools.reduce(lambda x, y: x*y, map(len, args))

        # Разделяем последовательность на части по индексам, потом сравниаем все соседние
        for cut_indexes in tqdm(
                itertools.product(*args),
                total=iter_count, desc='CHOW ', unit=' test'):
            try:
                cut_indexes = sorted(cut_indexes)
                test_sequence = []
                last_index = 0
                for index in cut_indexes:
                    test_sequence.append(self.sequence[last_index:index])
                    last_index = index
                test_sequence.append(self.sequence[last_index:])

                chow = 0
                for i in range(len(test_sequence)-1):
                    a = test_sequence[i]
                    b = test_sequence[i+1]
                    chow_test_result = chow_test.f_value(range(len(a)), a, range(len(a), len(a)+len(b)), b)
                    chow += chow_test_result[0]
                chow /= len(cut_indexes)+1
                if chow > max_chow_result:
                    max_chow_result = chow
                    max_chow_result_indexes = cut_indexes
            except IndexError:
                continue
        models = []
        last_index = 0
        for index in max_chow_result_indexes:
            models.append(Model(self.sequence[last_index:index]))
            last_index = index
        models.append(Model(self.sequence[last_index:]))
        if with_result:
            return max_chow_result, PartialModel(*models)
        return PartialModel(*models)


class PartialModel:
    def __init__(self, *models):
        self.models: List[Model] = list(models)

    def __getitem__(self, item):
        return self.models[item]

    def join(self):
        joined_models = np.array([])
        for model in self.models:
            joined_models = np.append(joined_models, model.sequence)
        return Model(joined_models)

    def sequence_for_draw(self, optimize=True):
        offset = 0
        last_value = None
        for model in self.models:
            if last_value is not None and optimize:
                value = np.append([np.nan] * (offset-1), np.append(last_value, model.sequence))
            else:
                value = np.append([np.nan] * offset, model.sequence)
            yield value
            last_value = value[-1]
            offset += len(model.sequence)

    def __getattr__(self, item):
        if item.startswith('each_'):
            return self._each(item[5:])
        if item.startswith('p') and item[1:].isdigit():
            return self.models[int(item[1:])-1]
        return super().__getattribute__(item)

    def _each(self, method_name):
        def method_call(*args, **kwargs):
            return tuple(getattr(model, method_name)(*args, **kwargs) for model in self.models)
        return method_call


class Arima:
    def __init__(self, endog, *args, **kwargs):
        if isinstance(endog, Model):
            endog = endog.sequence
        self.model = ARIMA(endog, *args, **kwargs)
        self.order = tuple(kwargs.get('order'))

    def fit(self, *args, **kwargs):
        _fitted = self.model.fit(*args, **kwargs)
        _fitted.sequence = _fitted.fittedvalues
        _fitted.seq = _fitted.sequence
        # FIXME: k-?
        _fitted.aic = Model(self.model.endog).aic(_fitted.sequence, self.order[0]+self.order[2])
        _fitted.order = self.order
        return _fitted

    @classmethod
    def find_optimal_model_by_order(cls, endog, p_sequence, d_sequence, q_sequence):
        if isinstance(endog, Model):
            endog = endog.sequence
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
