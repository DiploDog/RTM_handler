import cv2
from PIL import Image, ImageTk
import pathlib
from utils.profile_processor import ProfileReader
from main import NovatelParser
import numpy as np
import pandas as pd
import matplotlib.widgets as wdgt
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import (StringVar, IntVar, RIDGE, Canvas, Label, Entry, Button, Radiobutton, Tk, ttk)
from tkinter.messagebox import showinfo, showerror
from tkinter.filedialog import askopenfilename
from utils.profile_processor import PreStartPoint
from utils.ml import PolyRegression
from main import NovatelParser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
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


class App:

    version = 'v.0.0'
    img = 'logos/main_menu_pic.png'

    def __init__(self, root):
        self.root = root
        self.init_menu()

    def init_menu(self):
        self.root.title(f"Railway cars' resistance to movement handler ({self.version})")

        self.rp_default = 'data/best_profileBM.txt'
        self.railway_profile_var = StringVar()
        self.file_to_process = StringVar()
        self.kalman_var = StringVar()
        self.weight_state = IntVar()
        self.weight = StringVar()


        self.frame = ttk.Frame(self.root)
        self.frame.pack()
        self.frame.config(relief=RIDGE, padding=(15, 15))
        self.canvas = Canvas(self.frame, bg='white', width=600, height=400)
        self.display_img()
        self.canvas.pack()


        railway_profile_label = Label(self.root, text='Railway profile')
        rp_of_button = Button(self.root, text='Open file', command=self.profile_openfile)
        railway_profile_value = Entry(self.root, textvariable=self.railway_profile_var, width=60)
        railway_profile_value.insert(0, str(self.rp_default))

        experiment_label = Label(self.root, text='Experimental data file')
        exp_of_button = Button(self.root, text='Open file', command=self.process_openfile)
        experiment_value = Entry(self.root, textvariable=self.file_to_process, width=60)

        kalman_init_label = Label(self.root, text='Enter the initial acceleration value:')
        kalman_init = Entry(self.root, textvariable=self.kalman_var)
        kalman_init.insert(0, '0.0')

        rb1 = Radiobutton(self.root,
                          text='freight cond.',
                          value=1,
                          variable=self.weight_state,
                          command=self.weight_popup)
        rb2 = Radiobutton(self.root,
                          text='empty cond.',
                          value=2,
                          variable=self.weight_state,
                          command=self.weight_popup)

        ok_button = Button(self.root, text='Ok', command=self.go_to_process)

        railway_profile_label.pack()
        rp_of_button.pack()
        railway_profile_value.pack()
        experiment_label.pack()
        exp_of_button.pack()
        experiment_value.pack()
        kalman_init_label.pack()
        kalman_init.pack()
        rb1.pack(), rb2.pack()
        ok_button.pack()

    def weight_popup(self):
        weight_label = Label(self.root, text='Enter car weight, tons:')
        weight_entry = Entry(self.root, width=20)
        weight_label.pack(), weight_entry.pack()

    def process_openfile(self):
        curpath = pathlib.Path().resolve()
        filepath = askopenfilename(
            title='Select a file to process',
            initialdir=curpath,
            filetypes=[
                ('.data files', '.data')
            ]
        )
        self.file_to_process.set(filepath)

    def profile_openfile(self):
        curpath = pathlib.Path().resolve()
        filepath = askopenfilename(
            title='Select a profile file',
            initialdir=curpath,
            filetypes=[
                ('.txt files', '.txt')
            ]
        )
        self.railway_profile_var.set(filepath)

    def display_img(self):
        self.canvas.delete('all')
        try:
            image_read = cv2.imread(self.img)
            main_image = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB)
            h, w, _ = main_image.shape
            self.canvas.config(width=w, height=h)
            self.image_to_display = ImageTk.PhotoImage(Image.fromarray(main_image))
            self.canvas.create_image(w / 2, h / 2, image=self.image_to_display)
        except (FileNotFoundError, AttributeError, cv2.error):
            noimage_exception_label = Label(self.root, text='ImageLoader Error: No main menu image was found')
            noimage_exception_label.pack()

    def get_input(self):
        try:
            rp_in = self.railway_profile_var.get()
        except ValueError:
            pass
        try:
            proc_file_in = self.file_to_process.get()
        except ValueError:
            pass
        try:
            kalman_in = self.kalman_var.get()
        except ValueError:
            pass
        try:
            weight_in = self.weight.get()
        except ValueError:
            pass
        try:
            weight_state_in = self.weight_state.get()
        except ValueError:
            pass
        else:
            return rp_in, proc_file_in, kalman_in, weight_in, weight_state_in

    def go_to_process(self):
        profile, file, kalman_init, weight, weight_state = self.get_input()
        self.root.quit()
        run_handler(profile, file, kalman_init, weight, weight_state)


class Handler:

    def __init__(self, profile, file, kalman_init, weight, weight_state):
        self.profile = profile
        self.file = file
        self.kalman_init = kalman_init
        self.weight = weight
        self.weight_state = weight_state
        self.data = None

    def run(self):
        self.define_profile()
        pass

    def define_profile(self):
        self.profile = ProfileReader(self.profile).set_profile()

    def set_data(self):
        self.data = NovatelParser(self.file, 0).get_gprmc(fltr=False)

    def linearize_profile(self):
        # TODO: Понять надо ли вообще
        lr = LinearRegression()
        x = self.profile[['Широта', 'Долгота']]
        y = self.profile['Пикет']
        lr.fit(x, y)


    @staticmethod
    def ur_gruzh(col):
        return 0.4696 * col ** 2 - 0.2287 * col + 488.13

    @staticmethod
    def ur_por(col):
        return 0.5441 * col ** 2 - 2.1127 * col + 244.78

    class MplGUI:

        def __init__(self):
            pass

        class RangeCallback:







def run_handler(profile, file, kalman_init, weight, weight_state):



fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)
p, = plt.plot(arr['Время'], arr['Скорость'])
plt.xlabel('Время, с')
plt.ylabel('Cкорость, м/с')
plt.title('Зависимость скорости от времени')

# Create the RangeSlider
slider_ax = fig.add_axes([0.20, 0.2, 0.60, 0.03])
slider = wdgt.RangeSlider(ax=slider_ax,
                          label="Threshold",
                          valmin=arr['Время'].min(),
                          valmax=arr['Время'].max(),
                          valstep=0.2,
                          valinit=[arr['Время'].min() + 10, arr['Время'].max() - 10])


class Index:
    upper_limit, lower_limit = 0, 0

    def click_ok(self, event):
        self.upper_limit = upper_limit_line.get_data()[0][0]
        self.lower_limit = lower_limit_line.get_data()[0][0]
        ax.set_title('The threshold has been successfully set!', color='green')

    def get_limits(self):
        return self.lower_limit, self.upper_limit


callback = Index()

button_ax = fig.add_axes([0.75, 0.1, 0.2, 0.05])
button = wdgt.Button(ax=button_ax,
                     label='button',
                     color='grey',
                     hovercolor='green'
                     )

button.on_clicked(callback.click_ok)

# Create the Vertical lines on the histogram
lower_limit_line = ax.axvline(slider.val[0], color='k')
upper_limit_line = ax.axvline(slider.val[1], color='k')


def update(val):
    # The val passed to a callback by the RangeSlider will
    # be a tuple of (min, max)

    # Update the position of the vertical lines
    lower_limit_line.set_xdata([val[0], val[0]])
    upper_limit_line.set_xdata([val[1], val[1]])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()


slider.set_val([arr['Время'].min() + 10, arr['Время'].max() - 10])
slider.on_changed(update)

plt.show()

picet, slpe, df_len = NovatelParser.slopes_in_main(arr, prof)
arr = arr.copy().iloc[len(arr)-df_len:]
arr['Пикет'] = picet
arr['Уклон'] = slpe

lower_limit, upper_limit = callback.get_limits()

arr = arr[(arr['Время'] >= lower_limit) & (arr['Время'] < upper_limit)]
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

best_df['кв_скорости'] = best_df['Скорость'] ** 2
best_df['delta_v_qv'] = best_df['кв_скорости'].diff()

best_df['delta_h'] = best_df['Высота'].diff()
best_df.dropna(inplace=True)
best_df['Ускорение'] = ((best_df['delta_v_qv'] / 2) + (9.81 * best_df['delta_h'] / 1.06)) / best_df['Расстояние']
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
best_df.dropna(inplace=True)

best_df['Скорость'] = (best_df['Скорость'] * 3.6)
best_df['othcet'] = best_df['Скорость'].apply(ur_gruzh)
best_df.dropna(inplace=True)


best_df.to_excel('results' + file_name.split('.')[0] + '.xlsx')


if __name__ == '__main__':
    master = Tk()
    HelloWindow(master)
    master.mainloop()
