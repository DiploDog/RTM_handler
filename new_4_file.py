from utils.profile_processor import ProfileReader
from main import NovatelParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.profile_processor import PreStartPoint
from utils.ml import PolyRegression
from main import NovatelParser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import piecewise_regression
import filterpy
from filterpy import common, kalman
from geopy.distance import geodesic


def generate_degrees(source_data, degree):
    return np.array([
        source_data ** n for n in range(1, degree + 1)
    ]).T


def train_polynomial(x, y, xlabel, ylabel, degree, plot=True):
    try:
        X = generate_degrees(x, degree)
    except KeyError:
        print('No key for x in existing data!')
    try:
        if xlabel == 'Координаты':
            p, n, k = X.shape
            X = X.transpose(1, 0, 2).reshape(n, p*k)
        model = LinearRegression().fit(X, y)
    except KeyError:
        print('No key for y in existing data!')
    else:
        y_pred = model.predict(X)
        error = mse(y, y_pred)
        coeffs = model.coef_
        intercept = model.intercept_
        title = 'Степень полинома %d, Среднеквадратическая ошибка %.6f' \
                % (degree, error)

        if plot:
            if x.shape[-1] != 2:
                plt.scatter(x, y, 5, 'r', 'o', alpha=0.8, label='data')
                plt.plot(x, y_pred, c='b')
                plt.xlabel(xlabel + ', c')
                plt.ylabel(ylabel + ', м/с')
                plt.title(title)
                plt.show()

            else:
                plt.scatter(x.iloc[:, 0], y, 5, 'r', 'o', alpha=0.8, label='latitude')
                plt.plot(x.iloc[:, 0], y_pred, c='b', label='approx on lat')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
                plt.show()
                plt.scatter(x.iloc[:, 1], y, 5, 'g', 'o', alpha=0.8, label='longitude')
                plt.plot(x.iloc[:, 1], y_pred, c='black', label='approx on long')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
                plt.show()
                ax = plt.axes(projection='3d')
                ax.plot3D(x.iloc[:,0], x.iloc[:,1], y_pred)
                plt.show()

        return model, y_pred, error, coeffs, intercept

split = False
time_1 = 100
time_2 = 450
file_name = '0042.data'
file = ProfileReader('data/newest_profileBM.txt')
prof = file.set_profile()
ld = 1

lr = LinearRegression()
x = prof[['Широта', 'Долгота']]
y = prof['Пикет']
lr.fit(x, y)


data = NovatelParser('temp/' + file_name, 0)
arr = data.get_gprmc(fltr=False)


weight_gruzh = 100
def ur_gruzh(col):
    return 0.4696 * col**2 - 0.2287 * col + 488.13

weight_por = 25
def ur_por(col):
    return 0.5441 * col**2 - 2.1127 * col + 244.78


plt.plot(arr['Время'], arr['Скорость'])
plt.xlabel('Время, с')
plt.ylabel('Cкорость, м/с')
plt.title('Зависимость скорости от времени')
plt.show()
plt.savefig('temp/curves/' + file_name.split('.')[0] + 'v(t)_test' + '.jpg')
plt.clf()

picet, slpe, df_len = NovatelParser.slopes_in_main(arr, prof)
arr = arr.copy().iloc[len(arr)-df_len:]
arr['Пикет'] = picet
arr['Уклон'] = slpe



arr = arr[(arr['Время'] >= time_1) & (arr['Время'] < time_2)]
arr['Время'] = np.arange(0, arr.shape[0] * 0.2, 0.2)
plt.plot(arr['Время'], arr['Скорость'])
plt.xlabel('Время, с')
plt.ylabel('Cкорость, м/с')
plt.title('Зависимость скорости от времени')
plt.show()
plt.savefig('temp/curves/' + file_name.split('.')[0] + 'v(t)_test' + '.jpg')
plt.clf()


i = 3

arr['Пикет'] = lr.predict(arr[['Широта', 'Долгота']] * -1)

arr = arr[(abs(arr['Пикет'] % 50) >= 45) | (abs(arr['Пикет'] % 50) <= 5)]


arr['pic_diff'] = arr['Пикет'].diff()
arr.dropna(inplace=True)
arr['pic_cum'] = arr['pic_diff'].cumsum()
arr.dropna(inplace=True)
arr = arr.drop(columns=['Тег_данных',
                         'Корректность_данных',
                         'Широта',
                         'Долгота',
                         'Ориентация_широты',
                         'Ориентация_долготы',
                         'Дата',
                         'Уклон'])


#arr['Время'] = np.arange(0, arr.shape[0] * 0.2, 0.2)


def f(col1, col2):
    X = generate_degrees(col2, i)
    X = X.reshape((1, -1))
    res = (col1.predict(X)).item()
    return res


best_df = arr.copy()
best_df = best_df[best_df['Пикет'] < 9000]

crop_list = []
height_list = []
for i, pic_exp in enumerate(best_df['Пикет']):
    for pic_prof, height_prof in prof[['Пикет', 'Высота']].to_numpy():
        pic_diff = abs(pic_prof - pic_exp)
        if pic_diff < 3:
            crop_list.append(i)
            height_list.append(height_prof)

best_df = best_df.iloc[crop_list]
best_df['Высота'] = height_list


def ptr_gr(v):
    v = v * 3.6
    ptr = 9.81 * 100 * (0.7 + (3 + 0.09 * v + 0.002 * v ** 2) / 25)
    aero_multi = 0.0611 * v ** 2 + 0.8275 * v
    aero_solo = 0.4384 * v ** 2 - 0.2071 * v
    ptr_wko = ptr - aero_multi + aero_solo
    return ptr_wko


def ptr_por(v):
    v = v * 3.6
    ptr = 9.81 * 25 * (1.0 + 0.044 * v + 0.00021 * v ** 2)
    aero_multi = 0.0637 * v ** 2 + 1.2434 * v
    aero_solo = 0.5218 * v ** 2 - 0.6131 * v
    ptr_wko = ptr - aero_multi + aero_solo
    return ptr_wko


best_df['Wko_ptr'] = best_df['Скорость'].apply(ptr_gr)

best_df['Расстояние'] = best_df['Пикет'].diff()
best_df = best_df[best_df['Расстояние'] > 10]
best_df.dropna(inplace=True)

# model_v, _, _, coeffs_2, bias = train_polynomial(best_df['Время'], best_df['Скорость'], 'ХММММ', 'Скорость', 2, plot=True)
#
#
# best_df['Скорость'] = model_v.predict(generate_degrees(best_df['Время'], 2))


best_df['кв_скорости'] = best_df['Скорость'] ** 2
best_df['delta_v_qv'] = best_df['кв_скорости'].diff()

best_df['delta_h'] = best_df['Высота'].diff()
best_df.dropna(inplace=True)
best_df['left_part'] = best_df['delta_v_qv'] / 2 / best_df['Расстояние']
best_df['right_part'] = 9.81 * best_df['delta_h'] / 1.06 / best_df['Расстояние']
best_df['Ускорение'] = ((best_df['delta_v_qv'] / 2) + (9.81 * best_df['delta_h'] / 1.06)) / best_df['Расстояние']
#best_df['Ускорение'] = ((best_df['delta_v_qv'] / 2)) / best_df['Расстояние']


arr['Время'] = np.arange(0, arr.shape[0] * 0.2, 0.2)



dt = 0.2                     # Шаг времени
measurementSigma = 10          # Среднеквадратичное отклонение датчика
processNoise = 1e-4             # Погрешность модели

# Создаём объект KalmanFilter
filter = filterpy.kalman.KalmanFilter(dim_x=2,      # Размер вектора стостояния
                                     dim_z=1)      # Размер вектора измерений

# F - матрица процесса - размер dim_x на dim_x - 3х3
filter.F = np.array([[1, dt],
                     [0, 1]])


# Матрица наблюдения - dim_z на dim_x - 1x3
filter.H = np.array([[1.0, 0.0]])

# Ковариационная матрица ошибки модели
filter.Q = filterpy.common.Q_discrete_white_noise(dim=2, dt=dt, var=processNoise)

# Ковариационная матрица ошибки измерения - 1х1
filter.R = np.array([[measurementSigma*measurementSigma]])

# Начальное состояние.
filter.x = np.array([-0.08, 0.0])

# Ковариационная матрица для начального состояния
filter.P = np.array([[10.0, 0.0],
                    [0.0,  10.0]])

filteredState = []
stateCovarianceHistory = []

# Обработка данных
for i in range(0, best_df.shape[0]):
   z = [list(best_df['Ускорение'])[i]]                      # Вектор измерений
   filter.predict()                            # Этап предсказания
   filter.update(z)                            # Этап коррекции

   filteredState.append(filter.x)
   stateCovarianceHistory.append(filter.P)

filteredState = np.array(filteredState)
stateCovarianceHistory = np.array(stateCovarianceHistory)

# Визуализация
plt.title("Kalman filter (2nd order)")
plt.plot(best_df['Скорость'], best_df['Ускорение'], label="Измерение", color="#99AAFF")
plt.plot(best_df['Скорость'], filteredState[:, 0], label="Оценка фильтра", color="#224411")
plt.legend()
plt.show()

best_df['Несглаженное ускорение'] = best_df['Ускорение']
best_df['Ускорение'] = filteredState[:, 0]



best_df['Wko'] = - 1.06 * 1000 * best_df['Ускорение'] * weight_gruzh
#best_df = best_df[(best_df['Wko'] < 6000) & (best_df['Wko'] > 0)]
best_df.dropna(inplace=True)

best_df['Скорость'] = (best_df['Скорость'] * 3.6)
best_df['othcet'] = best_df['Скорость'].apply(ur_gruzh)
#best_df['parazit'] = best_df['Wko'] - best_df['othcet']
best_df.dropna(inplace=True)





best_df.to_excel('temp/curves/' + file_name.split('.')[0] + '_parts' + '.xlsx')

new_arr = pd.DataFrame()
new_arr[['Скорость', 'Wko']] = best_df.groupby(pd.cut(best_df['Скорость'], np.arange(0, 95, 5)))[
    ['Скорость', 'Wko']
].mean().round(1)
new_arr.dropna(inplace=True)

new_arr.to_excel('temp/curves/' + file_name.split('.')[0] + '_srednee_best' + '.xlsx')