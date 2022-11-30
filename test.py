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
split = False
time_1 = 125#125# 644 # 126.2 # 220 #100 # 231.2 #189 #702
time_2 = 465# 997 # 341.2 # 450 #475# 307.2 #295 #794.6
file_name = '0042.data'
file = ProfileReader('data/newest_profileBM.txt')
prof = file.set_profile()


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
        model = LinearRegression().fit(X, y)
    except KeyError:
        print('No key for y in existing data!')
    else:
        y_pred = model.predict(X)
        error = mse(y, y_pred)
        print(f'Коэффициенты: {model.coef_}, свободный член: {model.intercept_}')
        if plot:
            title = 'Степень полинома %d, Среднеквадратическая ошибка %.6f' \
                    % (degree, error)

            plt.scatter(x, y, 5, 'g', 'o', alpha=0.8, label='data')
            plt.plot(x, y_pred)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            # plt.ylim([0, -1])
            plt.title(title)
            plt.show()
        return y_pred, error

ld = 1

splitted = []
data = NovatelParser('temp/' + file_name, 0)
arr = data.get_gprmc(fltr=False)

if split:
    p_df_1 = arr[(arr['Время'] > 225) & (arr['Время'] < 309)]
    p_df_2 = arr[(arr['Время'] > 1198) & (arr['Время'] < 1288)]
    p_df_3 = arr[(arr['Время'] > 2295) & (arr['Время'] < 2384)]
    splitted.append(p_df_1)
    splitted.append(p_df_2)
    splitted.append(p_df_3)

    for i, arr in enumerate(splitted):
        picet, slpe, df_len = NovatelParser.slopes_in_main(arr, prof)
        arr = arr.copy().iloc[len(arr) - df_len:]
        arr['Пикет'] = picet
        arr['Уклон'] = slpe

        lat, lon = arr['Широта'].to_numpy() * -1, arr['Долгота'].to_numpy() * -1

        coords_zipIT = iter(zip(lat, lon))
        first = next(coords_zipIT)
        distnces = []
        try:
            while coords_zipIT:
                second = next(coords_zipIT)
                coord = PreStartPoint.geodistance(first, second)
                distnces.append(coord)
                first = second
        except StopIteration:
            pass

        distnces.insert(0, 0)
        arr['distance'] = distnces
        arr['cum_dist'] = arr['distance'].cumsum()
        arr.dropna(inplace=True)
        arr['Скорость'] = (arr['Скорость'] * 3.6)

        arr = arr[arr['Скорость'] > 0]
        arr = arr[arr['Скорость'].diff() < 0]

        plt.plot(arr['Время'], arr['Скорость'])
        plt.xlabel('Время, с')
        plt.ylabel('Cкорость, м/с')
        plt.title('Зависимость скорости от времени')
        plt.show()
        plt.clf()

        new_arr = pd.DataFrame()
        new_arr[['Пикет', 'Уклон', 'Скорость', 'Время']] = arr.groupby(pd.cut(arr['Время'], np.arange(0, 10000, 2)))[
            ['Пикет', 'Уклон', 'Скорость', 'Время']
        ].mean().round(1)
        new_arr['dv'] = new_arr['Скорость'].diff() / 3.6
        new_arr['dt'] = new_arr['Время'].diff()
        new_arr.dropna(inplace=True)
        new_arr['dv_dt'] = (new_arr['dv'] / new_arr['dt']).round(3)
        # arr = arr[arr['dv_dt'] > -0.1]
        new_arr['Wko'] = (-new_arr['Уклон'] - (1 + 0.06) * new_arr.dv_dt * 1000 / 9.81)

        new_arr.to_excel('new_arr_' + str(i) + '.xlsx')

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

#lat, lon = arr['Широта'].to_numpy() * -1, arr['Долгота'].to_numpy() * -1
lat, lon = arr['Широта'].to_numpy(), arr['Долгота'].to_numpy()

route_0042 = pd.DataFrame()
route_0042[['Широта', 'Долгота']] = arr[['Широта', 'Долгота']] * -1
route_0042[['Описание', 'Подпись']] = None
route_0042['Номер метки'] = np.arange(1, len(route_0042) + 1, 1)
route_0042.index = route_0042['Широта']
route_0042.drop(columns=['Широта'], inplace=True)
route_0042.to_csv('temp/curves/' + file_name.split('.')[0] + '_route' + '.csv', sep=',')


coords_zipIT = iter(zip(lat, lon))
first = next(coords_zipIT)
distnces = []
try:
    while coords_zipIT:
        second = next(coords_zipIT)
        coord = PreStartPoint.geodistance(first, second)
        distnces.append(coord)
        first = second
except StopIteration:
    pass

arr = arr[(arr['Время'] > time_1) & (arr['Время'] < time_2)]

train_polynomial(arr['Время'], arr['Скорость'], 'Время', 'Скорость', 3)

# distnces.insert(0, 0)
# arr['distance'] = distnces
# arr['cum_dist'] = arr['distance'].cumsum()
arr.dropna(inplace=True)
arr['Скорость'] = arr['Скорость'].round(1)
#arr = arr[arr['Скорость'].diff() > 0]
arr = arr[(arr['Время'] > time_1) & (arr['Время'] < time_2)]
arr['dv'] = arr['Скорость'].diff()
#arr = arr[(arr['dv'] >= -0.03)] #& (arr['dv'] < 0)]
arr['dt'] = arr['Время'].diff()
arr['dv_dt'] = arr['dv'] / arr['dt']
#arr = arr[arr['dv_dt'] > -0.1]

arr.dropna(inplace=True)
#arr = arr[arr['Скорость'] > 0]

arr['Wko'] = (-arr['Уклон'] - (1 + 0.06) * arr.dv_dt * 1000 / 9.81)








# TODO: Попробовать groupby пикетам и arr среднее или медиана

new_arr = pd.DataFrame()
new_arr[['Пикет', 'Уклон', 'Скорость', 'Время']] = arr.groupby(pd.cut(arr['Время'], np.arange(0, 10000, 10)))[
    ['Пикет', 'Уклон', 'Скорость', 'Время']
].mean().round(1)
new_arr['dv'] = new_arr['Скорость'].diff() / 3.6
new_arr['dt'] = new_arr['Время'].diff()
new_arr.dropna(inplace=True)
new_arr['dv_dt'] = (new_arr['dv'] / new_arr['dt']).round(3)
#arr = arr[arr['dv_dt'] > -0.1]
new_arr['Wko'] = (-new_arr['Уклон'] - (1 + 0.06) * new_arr.dv_dt * 1000 / 9.81)
#new_arr = new_arr[new_arr['Wko'] > 0]
# new_new = new_arr.groupby(pd.cut(new_arr['Скорость'], np.arange(0, 100, 2)))[['Скорость', 'Wko']].median().round(2)
# approx = train_polynomial(new_arr['Скорость'], new_arr['Wko'], 2)
# plt.plot(new_arr['Скорость'], approx)
# plt.show()
# new_arr = new_arr[(new_arr['Wko'] > 0.8 * approx) & (new_arr['Wko'] < 1.2 * approx)]


new_arr.to_excel('new_arr.xlsx')
#new_new.to_excel('new_new.xlsx')


arr_gr = arr.groupby('Скорость')['Wko'].mean()
arr_gr = arr_gr[2:]


arr.to_excel('temp/curves/' + file_name.split('.')[0] + 'v(t)_test2' + '.xlsx')
