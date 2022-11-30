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
time_1 = 180#100# 644 # 126.2 # 220 #100 # 231.2 #189 #702
time_2 = 385#997 #425# 997 # 341.2 # 450 #475# 307.2 #295 #794.6
file_name = '0042.data'
file = ProfileReader('data/newest_profileBM.txt')
prof = file.set_profile()
ld = 1

lr = LinearRegression()
x = prof[['Широта', 'Долгота']]
y = prof['Пикет']
lr.fit(x, y)


splitted = []
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



arr = arr[(arr['Время'] > time_1) & (arr['Время'] < time_2)]
arr['Время'] = np.arange(0, arr.shape[0] * 0.2, 0.2)
# arr = arr.copy().iloc[np.arange(0, arr.shape[0] + 1, 50)]
plt.plot(arr['Время'], arr['Скорость'])
plt.xlabel('Время, с')
plt.ylabel('Cкорость, м/с')
plt.title('Зависимость скорости от времени')
plt.show()
plt.savefig('temp/curves/' + file_name.split('.')[0] + 'v(t)_test' + '.jpg')
plt.clf()

j = 3
#model, _, _, coeffs, bias = train_polynomial(arr['Время'], arr['Скорость'], 'Время', 'Скорость', j, plot=True)

#print(prof[['Широта', 'Долгота']].to_numpy().reshape(-1, 2))

i = 2

#c, b, a = coeffs

print(prof.shape[0])
# TODO
# prof_errs = []
# prof_step = 4
# prof_cuts = []
# height_dict = {}
# prof['Модель'] = 0
# for i_part in range(0, prof.shape[0], prof_step):
#     prof_cut_val = i_part + prof_step
#     small_prof = prof.copy().iloc[i_part:prof_cut_val, :]
#     model_c, _, errorrr, coeffs_c, bias_c = train_polynomial(small_prof['Пикет'].values,
#                                           small_prof['Высота'].values, 'Пикет', 'Высота', i, plot=False)
#     small_prof['Высота'] = model_c.predict(generate_degrees(small_prof['Пикет'], i))
#     small_prof['Модель'] = model_c
#     height_dict[tuple(small_prof['Пикет'])] = model_c
#     prof_cuts.append(small_prof)
# prof_df = pd.concat(prof_cuts)
# prof_err = mse(prof['Высота'], prof_df['Высота'])
#
# best_df = prof_df
# prof['Высота_расчет'] = best_df['Высота']
# prof['Ошибка_абс'] = (prof['Высота_расчет'] - prof['Высота']).abs()
# prof['model'] = best_df['Модель']
# prof.dropna(inplace=True)
# intervals = list((pd.cut(prof['Пикет'], np.arange(2500, 9000, 200))).unique())[1:]
# df_intervals = []
# for i_picet in prof['Пикет']:
#     for i_interval in intervals:
#         if i_picet in i_interval:
#             df_intervals.append(i_interval)
#             break
# prof = prof.iloc[:prof.shape[0]-1, :]
# prof['picet_cut'] = df_intervals
# prof['Коэффициенты'] = prof.model.apply(lambda x: x.coef_)
# prof['lower_val'] = prof['picet_cut'].apply(lambda x: x.left)
# prof['upper_val'] = prof['picet_cut'].apply(lambda x: x.right)
# #print(prof)
# prof.to_excel('profile_approx.xlsx')

# model_c, _, errorrr, coeffs_c, bias_c = train_polynomial(prof['Пикет'],
#                                       prof['Высота'], 'Пикет', 'Высота', i, plot=True)
# TODO

models_list = []
coeffs_list = []
picets = []


def uravnenie_3(col):
    return a * col**3 + b * col ** 2 + c * col + bias


def uravnenie_2(col):
    return q * col ** 2 + w * col + bias

#
# def uravnenie_c(col):
#     return d * col**3 + e * col ** 2 + f * col + bias_c


# rows = []
# heights = []
# pics = []
# slps = []
# for i, shir_dol in enumerate(arr[['Широта', 'Долгота']].to_numpy()):
#     shirota, dolgota = shir_dol.round(5)
#     for true_shir, true_lon in prof[['Широта', 'Долгота']].to_numpy():
#         diff_lat = abs(round(shirota * -1, 5) - true_shir)
#         diff_lon = abs(round(dolgota * -1, 5) - true_lon)
#         if diff_lat < 0.00001 and diff_lon < 0.00001:
#             rows.append(i)
#
# arr = arr.copy().iloc[rows]

arr['Пикет'] = lr.predict(arr[['Широта', 'Долгота']] * -1)
#arr['Скорость'] = arr['Скорость'] * 3.6
#arr.to_excel('initial_dataframe.xlsx')

# for i_picet in arr['Пикет']:
#     for left_pic, right_pic, i_model, i_coeffs, i_pc in prof[['lower_val', 'upper_val', 'model', 'Коэффициенты', 'picet_cut']].to_numpy():
#         if left_pic <= i_picet < right_pic:
#             models_list.append(i_model)
#             coeffs_list.append(i_coeffs)
#             picets.append(i_pc)
#             break
#
# arr['Модель'] = models_list
# arr['Коэффициенты'] = coeffs_list
# arr['Срез_Пикетов'] = picets


def f(col1, col2):
    X = generate_degrees(col2, i)
    X = X.reshape((1, -1))
    res = (col1.predict(X)).item()
    return res

#TODO
"""print(arr)
dfs = []
step = 0
errs = []
p = 1
speed_mses = []
for i_step in range(20, arr.shape[0], 20):
    cuts = []
    step = i_step
    speed_mse = []
    # prof_chunk = 4
    # init_prof = 0
    print('Итерация:', p)
    for init_cut in range(0, arr.shape[0], step):
        cut_val = init_cut + step
        small_arr = arr.copy().iloc[init_cut:cut_val, :]
        #small_arr = small_arr.copy().iloc[np.arange(0, small_arr.shape[0], 50)]
        small_arr['Старая скорость'] = small_arr['Скорость']

        small_model_v, _, _, small_coeffs_2, small_bias = train_polynomial(
            small_arr['Пикет'].values, small_arr['Скорость'].values, 'Пикет', 'Скорость', 1, plot=False)"""
#TODO


        # prof_val = init_prof + prof_chunk
        # cut_prof = prof.copy().iloc[init_prof:prof_val, :]
        # init_prof = prof_val
        # prof_chunk += 5
        # print(prof_chunk)

        # model_c, _, errorrr, coeffs_c, bias_c = train_polynomial(cut_prof['Пикет'],
        #                                                          cut_prof['Высота'], 'Пикет', 'Высота', i, plot=False)

        #small_arr['Расстояние'] = small_arr['Пикет'].diff()
        # TODO
        #data_v = generate_degrees(small_arr['Пикет'].values, 1)
        #small_arr['Скорость'] = small_model_v.predict(data_v)
        # TODO
        # data = generate_degrees(small_arr['Пикет'], i)
        # small_arr['Высота'] = model_c.predict(data)
        # small_arr['Высота'] = small_arr.apply(lambda x: f(x['Модель'], x['Пикет']), axis=1)
        # small_arr = small_arr[small_arr['Высота'].diff() > 0]

        # small_arr['кв_скорости'] = small_arr['Скорость'] ** 2
        # small_arr['delta_v_qv'] = small_arr['кв_скорости'].diff()
        # small_arr['delta_h'] = small_arr['Высота'].diff()
        # small_arr['left_part'] = small_arr['delta_v_qv'] / 2 / small_arr['Расстояние']
        # small_arr['right_part'] = 9.81 * small_arr['delta_h'] / 1.06 / small_arr['Расстояние']
        # small_arr['Ускорение'] = (small_arr['delta_v_qv'] / 2 + 9.81 * small_arr['delta_h'] / 1.06) / small_arr['Расстояние']
        # small_arr['Wko'] = - 1.06 * 1000 * small_arr['Ускорение'] * weight_gruzh
        # #small_arr = small_arr[small_arr['Wko'] > 0]
        # #small_arr = small_arr[small_arr['Wko'] > 0]
        # small_arr['Скорость'] = (small_arr['Скорость'] * 3.6)
        # small_arr['othcet'] = small_arr['Скорость'].apply(ur_gruzh)
        # small_arr.dropna(inplace=True)

"""        cuts.append(small_arr)

    part_dif = pd.concat(cuts)
    dfs.append(part_dif)
    # err = mse(part_dif['othcet'].to_numpy(), part_dif['Wko'].to_numpy())
    # errs.append(err)
    p += 1
    err_speed = mse(part_dif['Старая скорость'].to_numpy(), part_dif['Скорость'].to_numpy())
    speed_mses.append(err_speed)

# best_mse = np.argmin(errs)
# best_df = dfs[best_mse]
best_speed_mse = np.argmin(speed_mses)
best_df = dfs[best_speed_mse]
print(len(dfs))"""
best_df = arr.copy()
#best_df = best_df[best_df['Пикет'] < 7500]
#best_df = best_df[abs(best_df['Пикет'] // 100 * 100 - best_df['Пикет']) < 3]
#best_df = best_df.copy().iloc[np.arange(0, best_df.shape[0], 50)]
#print(f'Лучшая МСЕ: {errs[best_mse]}, по счету она {best_mse}')
#print(f'Лучшая МСЕ по скорости: {speed_mses[best_speed_mse]}, по счету она {best_speed_mse}')
#best_df = arr.copy()
#best_df = best_df[(best_df['Скорость'] < 25) & (best_df['Скорость'] > 5)]
best_df['Расстояние'] = best_df['Пикет'].diff()

best_df.dropna(inplace=True)

model_rasstoyanie, _, _, _, _ = train_polynomial(best_df['Время'], best_df['Расстояние'], 'Скорость', 'Расстояние', 3, plot=False)
best_df['Расстояние'] = model_rasstoyanie.predict(generate_degrees(best_df['Время'], 3))

model_v, _, _, coeffs_2, bias = train_polynomial(best_df['Время'], best_df['Скорость'], 'ХММММ', 'Скорость', 2, plot=True)
model_c, _, _, coeffs_c, bias_c = train_polynomial(prof['Пикет'], prof['Высота'], 'Пикет', 'Высота', 1, plot=False)

best_df['Высота'] = model_c.predict(generate_degrees(best_df['Пикет'], 1))
best_df['Скорость'] = model_v.predict(generate_degrees(best_df['Время'], 2))


#best_df['Высота'] = best_df.apply(lambda x: f(x['Модель'], x['Пикет']), axis=1)
best_df['кв_скорости'] = best_df['Скорость'] ** 2
best_df['delta_v_qv'] = best_df['кв_скорости'].diff()

best_df['delta_h'] = best_df['Высота'].diff()
best_df.dropna(inplace=True)
print(best_df.shape)
best_df['left_part'] = best_df['delta_v_qv'] / 2 / best_df['Расстояние']
best_df['right_part'] = 9.81 * best_df['delta_h'] / 1.06 / best_df['Расстояние']
best_df['Ускорение'] = ((best_df['delta_v_qv'] / 2) + (9.81 * best_df['delta_h'] / 1.06)) / best_df['Расстояние']

best_df['Wko'] = - 1.06 * 1000 * best_df['Ускорение'] * weight_por
#best_df = best_df[(best_df['Wko'] < 10000) & (best_df['Wko'] > 0)]
best_df.dropna(inplace=True)
# model_wko, _, _, _, _ = train_polynomial(best_df['Скорость'], best_df['Wko'], 'Скорость', 'Wko', 2, plot=False)
# best_df = best_df[(best_df['Wko'] < model_wko.predict(generate_degrees(best_df['Скорость'], 2)) + 300) &
#                   (best_df['Wko'] > model_wko.predict(generate_degrees(best_df['Скорость'], 2)) - 300)]

best_df['Скорость'] = (best_df['Скорость'] * 3.6)
best_df['othcet'] = best_df['Скорость'].apply(ur_gruzh)
best_df['parazit'] = best_df['Wko'] - best_df['othcet']
best_df.dropna(inplace=True)
print(best_df.shape)





# new_df = pd.DataFrame()
# new_df['Время'] = np.linspace(0, 450, 2250).round(1)
# new_df['Скорость'] = model_v.predict(generate_degrees(new_df['Время'], 2))
# new_df['Расстояние'] = model_rasstoyanie.predict(generate_degrees(new_df['Время'], 2))
# #new_df['Высота'] = model_c.predict(generate_degrees(new_df['Время'], 1))
# new_df['кв_скорости'] = new_df['Скорость'] ** 2
# new_df['delta_v_qv'] = new_df['кв_скорости'].diff()
# #new_df['delta_h'] = new_df['Высота'].diff()
# new_df['Ускорение'] = (new_df['delta_v_qv'] / 2) / new_df['Расстояние'] - 0.03186
# new_df['Wko'] = - 1.06 * 1000 * new_df['Ускорение'] * weight_gruzh
# new_df['Скорость'] = (new_df['Скорость'] * 3.6)
# new_df['othcet'] = new_df['Скорость'].apply(ur_gruzh)
# new_df.to_excel('new_df.xlsx')



#part_dif = pd.concat(cuts)
best_df.to_excel('temp/curves/' + file_name.split('.')[0] + '_parts' + '.xlsx')

new_arr = pd.DataFrame()
new_arr[['Скорость', 'Wko']] = best_df.groupby(pd.cut(best_df['Скорость'], np.arange(0, 92, 2)))[
    ['Скорость', 'Wko']
].mean().round(1)
new_arr.dropna(inplace=True)

new_arr.to_excel('temp/curves/' + file_name.split('.')[0] + '_srednee_best' + '.xlsx')