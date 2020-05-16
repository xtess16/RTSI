import functools
import itertools
import time
import traceback
from multiprocessing.pool import Pool
from typing import Union, Sequence, Optional, Tuple, List

import RegscorePy as rp
import chow_test
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels.api as sm
from arch import arch_model
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf, acf
from tqdm import tqdm
import pandas as pd

T_SERIES = Union[pd.Series, 'Model']


def draw(*args, **kwargs):
    """ plt.plot специально для рисования Model и PartialModel"""
    result_args = []
    for arg in args:
        if isinstance(arg, Model):
            result_args.append(arg.series)
        elif isinstance(arg, PartialModel):
            for s in arg.draw_series():
                draw(s, **kwargs)
            return
        else:
            result_args.append(arg)
    return plt.plot(*result_args, **kwargs)


class Model:
    def __init__(self, series: T_SERIES, k=None):
        if isinstance(series, PartialModel):
            series = series.join()
        elif not isinstance(series, pd.Series):
            raise TypeError('Аргумент series может быть только PartialModel или pd.Series')
        self.series: pd.Series = series
        self.k = k

    def __getattr__(self, item):
        res = getattr(self.series, item)
        if isinstance(res, pd.Series):
            return Model(res)
        return res

    def __getitem__(self, item):
        return Model(self.series[item])

    def __iter__(self):
        return iter(self.series)

    def __sub__(self, other):
        if isinstance(other, Model):
            return Model(self.series - other.series)
        return Model(self.series - other)

    def __add__(self, other):
        if isinstance(other, Model):
            return Model(self.series + other.series)
        return Model(self.series + other)

    def __mul__(self, other):
        if isinstance(other, Model):
            return Model(self.series * other.series)
        return Model(self.series * other)

    def __pow__(self, power):
        return Model(self.series ** power)

    def __truediv__(self, other):
        if isinstance(other, Model):
            return Model(self.series / other.series)
        return Model(self.series / other)

    def __floordiv__(self, other):
        if isinstance(other, Model):
            return Model(self.series // other.series)
        return Model(self.series // other)

    def __len__(self):
        return len(self.series)

    def append(self, other, inplace=False):
        """
            Склеивает последовательности
        :param other: Последовательность, которая будет приклеена
        :param inplace: Если True, то изменит список текущей модели, False - вернет новый
        :return: Склеенный список
        """
        if inplace:
            self.series = self.series.append(other.series)
        else:
            return Model(self.series.append(other.series))

    def fuller_test(self):
        """ Тест Фуллера """
        return adfuller(self.series)

    def durbin_watson_test(self):
        """ Тест Дарбина Уотсона """
        dw = 0
        for t in range(1, len(self.series)):
            dw += (self.series[t] - self.series[t-1]) ** 2
        dw /= (self.series ** 2).sum()
        return dw

    def arima(self, *args, **kwargs):
        """ Построение модели АРИМА """
        return Arima(self.series, *args, **kwargs)

    def acf(self, lag: int, **kwargs):
        """
            Поиск автокорреляции
        :param lag: Значение лага
        :return: Список/Значение корреляции
        """
        kwargs['nlags'] = lag
        if kwargs.get('qstat', False):
            return acf(self, **kwargs)
        return acf(self, **kwargs)[lag-1]

    def many_acf(self, lag_from: int, lag_to: int, **kwargs):
        """
            Поиск автокорреляции по нескольким лагам
        :param lag_from: Поиск ОТ лага
        :param lag_to: Поиск ДО лага
        :return: Список/Значение корреляции
        """
        return np.array([self.acf(i, **kwargs) for i in range(lag_from, lag_to)])

    def pacf(self, lag: int):
        """
            Поиск частной автокорреляции
        :param lag: Значение лага
        :return: Список/Значение корреляции
        """
        return pacf(self)[lag-1]

    def many_pacf(self, lag_from: int, lag_to: int):
        """
            Поиск частной автокорреляции по нескольким лагам
        :param lag_from: Поиск ОТ лага
        :param lag_to: Поиск ДО лага
        :return: Список/Значение корреляции
        """
        return np.array([self.pacf(i) for i in range(lag_from, lag_to)])

    def aic(self, array: T_SERIES, k: Optional[int] = None) -> float:
        """
            Поиск КФ Акаике
        :param array: Второй список
        :param k: Количество параметров
        :return: КФ Акаике
        """
        if isinstance(array, Model):
            if k is None:
                k = array.k
            array = array.series.tolist()
        elif isinstance(array, np.ndarray):
            array = array.tolist()
        return rp.aic.aic(self.series.tolist(), array, k) / len(self.series)

    def bic(self, array: T_SERIES, k: Optional[int] = None):
        """
            Вычисление критерия Шварца
        :param array: Второй список
        :param k: Количество параметров
        :return: Критерий Шварца
        """
        if isinstance(array, Model):
            if k is None:
                k = array.k
            array = array.series.tolist()
        elif isinstance(array, np.ndarray):
            array = array.tolist()
        return rp.bic.bic(self.series.tolist(), array, k) / len(self.series)

    def polynomial_trend(self, degree):
        """ Построение тренда по степени полинома """
        pf = PolynomialFeatures(degree=degree)
        xp = pf.fit_transform(np.arange(1, self.series.size+1)[:, np.newaxis])
        fitted_model = sm.OLS(self.series, xp).fit()
        trend_line = Model(fitted_model.fittedvalues)
        trend_line.fitted_model = fitted_model
        return trend_line

    def polyfit(self, k: Optional[int] = None):
        """ Параметры полинома """
        if k is None:
            k = self.k
        p = np.polyfit(range(len(self.series)), self.series, k)
        return p

    def determination_kf(self, k: Optional[int] = None) -> float:
        """
            Подсчет КФ детерминации
        :param k: Количество параметров модели
        :return: КФ детерминации
        """
        if k is None:
            k = self.k
        pl = self.trend(self.polyfit(k)).series
        mean = self.series.mean()
        top = ((pl - mean) ** 2).sum()
        bottom = ((self.series - mean) ** 2).sum()
        determination = top / bottom
        return determination

    def partial_determination_kf(self, other) -> float:
        """
            Подсчет КФ детерминации
        :param other: Кусочный тренд
        :return: КФ детерминации
        """
        pl = other.series
        mean = self.series.mean()
        top = ((pl - mean) ** 2).sum()
        bottom = ((self.series - mean) ** 2).sum()
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
        return (self.determination_kf(k) / (k-1)) / ((1 - self.determination_kf(k)) / (len(self.series) - k))

    def adjusted_determination_kf(self, k: Optional[int] = None):
        """
            Подсчет скорректированного КФ детерминации
        :param k: Количество параметров модели
        :return: Скорректированный КФ детерминации
        """
        if k is None:
            k = self.k
        determination_kf = self.determination_kf(k)
        kf = 1 - (1 - determination_kf) * ((len(self.series) - 1) / (len(self.series) - k))
        return kf

    def exponential_smoothing(self, alpha):
        """
            Экспоненциальное сглаживание
        :param alpha: КФ
        :return:
        """
        # TODO: найти либу
        # XXX
        result = self.series.copy()
        for n in range(1, len(result)):
            result[n] = alpha * result[n] + (1 - alpha) * result[n - 1]
        return Model(result)

    def cut_by_chow_test(self, *args, arbitrarily: Optional[int] = None, with_result=False):
        """
            Поиск оптимального разделения графика по Чоу тесту
        :param args: Набор диапазонов в которых будет происходить точек для деления
        :param arbitrarily: Если передано число,
            то метод сам разделит последовательность на arbitrarily частей
        :param with_result: Возвращать результат теста или нет
        :return: Если with_result - True, то возвращает кортеж из 3 элементов:
            Значение лучшего теста, список индексов, делящих series на части, PartialModel
        """
        # Если передано arbitrarily, то перебираем все значения
        if isinstance(arbitrarily, int):
            args = tuple(range(len(self.series)) for _ in range(arbitrarily-1))
        max_chow_result: float = 0
        max_chow_result_indexes: Optional[List[int]] = None
        if isinstance(args[0], slice):
            iter_count = functools.reduce(lambda x, y: x * y, map(lambda z: len(self.series[z]), args))
            args = [self.series[s].index for s in args]
        else:
            iter_count = functools.reduce(lambda x, y: x*y, map(len, args))

        # Разделяем последовательность на части по индексам, потом сравниаем все соседние
        for cut_indexes in tqdm(
                itertools.product(*args),
                total=iter_count, desc='CHOW ', unit=' test'):
            try:
                cut_indexes = sorted(cut_indexes)
                test_series = []
                last_index = 0
                for index in cut_indexes:
                    test_series.append(self.series[last_index:index])
                    last_index = index
                test_series.append(self.series[last_index:])

                chow = 0
                for i in range(len(test_series)-1):
                    a = test_series[i]
                    b = test_series[i+1]
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
            models.append(Model(self.series[last_index:index]))
            last_index = index
        models.append(Model(self.series[last_index:]))
        if with_result:
            return max_chow_result, [self.series.index[i] for i in max_chow_result_indexes], PartialModel(*models)
        return PartialModel(*models)

    def barlett(self):
        """ Критерий Барлетта """
        return scipy.stats.bartlett(self.series, range(len(self.series)))

    def levene(self):
        """ Критерий Левене """
        return scipy.stats.levene(self.series, range(len(self.series)))

    def mape(self, other):
        """ Критерий MAPE """
        return np.mean(np.abs((self - other) / self)) * 100

    def mae(self, other):
        """ Критерий MAE """
        return np.mean(np.abs(self - other))


class PartialModel:
    def __init__(self, *models):
        models = [model if isinstance(model, Model) else Model(model) for model in models]
        self.models: List[Model] = models

    def __getitem__(self, item):
        return self.models[item]

    def __iter__(self):
        return iter(self.models)

    def __add__(self, other):
        return PartialModel(*[item1 + item2 for item1, item2 in zip(self, other)])

    def __sub__(self, other):
        return PartialModel(*[item1 - item2 for item1, item2 in zip(self, other)])

    def __mul__(self, other):
        return PartialModel(*[item1 * item2 for item1, item2 in zip(self, other)])

    def __truediv__(self, other):
        return PartialModel(*[item1 / item2 for item1, item2 in zip(self, other)])

    def __floordiv__(self, other):
        return PartialModel(*[item1 // item2 for item1, item2 in zip(self, other)])

    def __pow__(self, power):
        return PartialModel(*[item ** power for item in self])

    def join(self):
        """ Объединение всех кусков в одну модель """
        joined_models = pd.Series()
        for model in self.models:
            joined_models = joined_models.append(model.series)
        return Model(joined_models)

    def draw_series(self):
        """
            Последовательность self.models, где у каждого model в конце добавляется первое
        значение следующего model
        """
        res = [self.models[i].append(self.models[i+1][:1]) for i in range(len(self.models)-1)]
        res.append(self.models[-1])
        return PartialModel(*res)

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
            endog = endog.series
        self.model = ARIMA(endog, *args, **kwargs)
        self.order = tuple(kwargs.get('order'))

    def fit(self, *args, **kwargs):
        """ Тренировка модели """
        _fitted = self.model.fit(*args, **kwargs)
        _fitted.order = self.order
        return _fitted

    @classmethod
    def find_optimal_model_by_order(cls, endog, p_series, d_series, q_series, top=1):
        """
            Подбор оптимальных p, d, q для временного ряда
        :param endog: Последовательность
        :param p_series: параметр p
        :param d_series: параметр d
        :param q_series: параметр q
        :param top: Количество лучших моделей, которое надо вернуть
        :return: Натренированная модель с наименьшим aic
        """
        if isinstance(endog, Model):
            endog = endog.series
        if isinstance(p_series, int):
            p_series = (p_series, )
        if isinstance(d_series, int):
            d_series = (d_series, )
        if isinstance(q_series, int):
            q_series = (q_series, )

        start = time.monotonic()
        pool = Pool()
        result = pool.map(
            cls.pdq_handler,
            [(endog, p, d, q) for p, d, q in itertools.product(p_series, d_series, q_series)]
        )
        finish = time.monotonic() - start
        print(f'Перебрал {len(result)} вариант(а/ов) за {finish:.2f} сек.')
        if top == 1:
            min_fitted_model = min(result, key=lambda x: float('inf') if x is None else x.aic)
            return min_fitted_model
        else:
            results = []
            for _ in range(top):
                if not result:
                    results.append(None)
                else:
                    tmp_min = min(result, key=lambda x: float('inf') if x is None else x.aic)
                    result.remove(tmp_min)
                    results.append(tmp_min)
            return results

    @staticmethod
    def pdq_handler(args):
        """ Возвращает обученную на переданных данных модель """
        try:
            _model = Arima(args[0], order=args[1:])
            _fitted = _model.fit()
        except KeyboardInterrupt:
            return None
        except ValueError as e:
            if 'pass your own start_params.' in str(e):
                return None
            elif 'On entry to DLASCL parameter number 4 had an illegal value' in str(e):
                return None
            print(args[1:])
            print(traceback.format_exc())
        except np.linalg.LinAlgError:
            pass
        except Exception as e:
            if 'Expect positive integer' not in str(e):
                print(args[1:])
                print(traceback.format_exc())
        else:
            return _fitted

    def __getattr__(self, item):
        return getattr(self.model, item)


arch_model