from utils.profile_processor import ProfileReader
from main import NovatelParser
from utils.profile_processor import PreStartPoint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import numpy as np

file_name = '0063.data'
file = ProfileReader('data/profileBM.txt')
df = file.profile_df
new_df = file.shift_df()
dtfrm = file.widen_dfs()
slope = file.calculate_slope()
predicted = file._get_widened_df()
#print(predicted)
file = NovatelParser('temp/' + file_name)
arr = file.get_gprmc()
arr = arr[arr['Время'] < 50.0]
print(arr)
#print(arr)



def slopes_in_main(arr, help):
    pic = []
    slp = []
    for lat, lon in zip(arr['Широта'] * -1, arr['Долгота'] * -1):
        for lat1, lon1, lat2, lon2, pc, sl in help[['Широта',
                                                    'Долгота',
                                                    'Широта_след.',
                                                    'Долгота_след.',
                                                    'Пикет',
                                                    'Уклон']].to_numpy():

            if lat1 >= lat >= lat2:
                pic.append(pc)
                slp.append(sl)
    return pic, slp


picet, slpe = slopes_in_main(arr, predicted)
arr['Пикет'] = picet[:-1]
arr['Уклон'] = slpe[:-1]

arr['dv_dt'] = arr['Скорость'].diff() / arr['Время'].diff()

arr.fillna(arr['dv_dt'][1], inplace=True)

print(arr.head(30))


def generate_degrees(source_data, degree):
    return np.array([
        source_data**n for n in range(1, degree + 1)
    ]).T


def train_polynomial(degree, data):
    X = generate_degrees(data['Время'], degree)

    model = LinearRegression().fit(X, data['dv_dt'])
    y_pred = model.predict(X)

    error = mse(data['dv_dt'], y_pred)
    title = 'Степень полинома %d, Среднквадратическая ошибка %.3f' % (degree, error)

    plt.scatter(data['Время'], data['dv_dt'], 5, 'g', 'o', alpha=0.8, label='data')
    plt.plot(data['Время'], y_pred)
    plt.xlabel('Время, с')
    plt.ylabel('Ускорение, м/(с*с)')
    plt.title(title)
    plt.savefig('temp/curves/' + file_name.split('.')[0] + '_poly_dv_dt' + '.jpg')
    plt.close()


train_polynomial(2, arr)

arr['Wko'] = (25 * 9.81 * arr['Уклон'] + (25 * (1 + 1.06) * arr.dv_dt)) / 25

plt.plot(arr['Время'], arr['Скорость'])
plt.xlabel('Время, с')
plt.ylabel('Cкорость, м/с')
plt.title('Зависимость скорости от времени')
plt.savefig('temp/curves/' + file_name.split('.')[0] + 'v(t)' + '.jpg')

arr.to_excel('test_arr_64.xlsx')

# Index(['Тег данных', 'Время, с', 'Корректность данных', 'Широта',
#        'Ориентация широты', 'Долгота', 'Ориентация долготы', 'Скорость, м/с',
#        'Дата', 'Пикет', 'Уклон'],
#       dtype='object')

