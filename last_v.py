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
        weight_entry = Entry(self.root, textvariable=self.weight, width=20)
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
            kalman_in = float(kalman_in)
        except ValueError:
            pass
        try:
            weight_in = self.weight.get()
            weight_in = float(weight_in)
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
        run_handler(profile, file, self.file_to_process, kalman_init, weight, weight_state)


class Handler:

    def __init__(self, profile, file, file_name, kalman_init, weight, weight_state):
        self.profile = profile
        self.file = file
        self.file_name = file_name
        self.kalman_init = kalman_init
        self.weight = weight
        self.weight_state = weight_state
        self.data = None
        self.lower_limit = None
        self.upper_limit = None

    def run(self):
        self.define_profile()
        self.set_data()
        self.linearize_profile()
        self.set_limits()
        self.set_railway_params()
        self.cut_on_limits()
        self.plot_cut()
        self.get_data_on_measured_railway()
        self.calc_n_drop()
        self.heights_and_params()
        self.ptr_comparison()
        self.calculate_values()
        self.kalman_filter()
        self.finishing_touches()
        self.save_results()

    def define_profile(self):
        self.profile = ProfileReader(self.profile).set_profile()

    def set_data(self):
        self.data = NovatelParser(self.file, 0).get_gprmc(fltr=False)

    def linearize_profile(self):
        lr = LinearRegression()
        x = self.profile[['Широта', 'Долгота']]
        y = self.profile['Пикет']
        lr.fit(x, y)
        self.data['Пикет'] = lr.predict(self.data[['Широта', 'Долгота']] * -1)

    @staticmethod
    def ur_gruzh(v):
        return 0.4696 * v ** 2 - 0.2287 * v + 488.13

    @staticmethod
    def ur_por(v):
        return 0.5441 * v ** 2 - 2.1127 * v + 244.78

    @staticmethod
    def ptr_gr(v):
        v = v * 3.6
        ptr = 9.81 * 100 * (0.7 + (3 + 0.09 * v + 0.002 * v ** 2) / 25)
        aero_multi = 0.0611 * v ** 2 + 0.8275 * v
        aero_solo = 0.4384 * v ** 2 - 0.2071 * v
        ptr_wko = ptr - aero_multi + aero_solo
        return ptr_wko

    @staticmethod
    def ptr_por(v):
        v = v * 3.6
        ptr = 9.81 * 25 * (1.0 + 0.044 * v + 0.00021 * v ** 2)
        aero_multi = 0.0637 * v ** 2 + 1.2434 * v
        aero_solo = 0.5218 * v ** 2 - 0.6131 * v
        ptr_wko = ptr - aero_multi + aero_solo
        return ptr_wko

    def set_limits(self):
        mplgui_object = self.MplGUI(self.data)
        self.lower_limit, self.upper_limit = mplgui_object.get_result()

    class MplGUI:

        def __init__(self, data):
            self.data = data
            self.fig, self.ax = None, None
            self.lower_limit_line = None
            self.upper_limit_line = None
            self.slider = None
            self.button = None

            self.init_plot()

            self.create_widgets()

            self.callback = self.RangeCallback(self.ax,
                                               self.lower_limit_line,
                                               self.upper_limit_line)
            self.button.on_clicked(self.callback.click_ok)

            plt.show()

        def init_plot(self):
            self.fig, self.ax = plt.subplots()
            plt.subplots_adjust(left=0.1, bottom=0.35)
            p, = plt.plot(self.data['Время'], self.data['Скорость'])
            plt.xlabel('Время, с')
            plt.ylabel('Cкорость, м/с')
            plt.title('Зависимость скорости от времени')

        def create_widgets(self):

            slider_ax = self.fig.add_axes([0.20, 0.2, 0.60, 0.03])
            self.slider = wdgt.RangeSlider(ax=slider_ax,
                                      label="Threshold",
                                      valmin=self.data['Время'].min(),
                                      valmax=self.data['Время'].max(),
                                      valstep=0.2,
                                      valinit=[self.data['Время'].min() + 10, self.data['Время'].max() - 10])

            self.lower_limit_line = self.ax.axvline(self.slider.val[0], color='k')
            self.upper_limit_line = self.ax.axvline(self.slider.val[1], color='k')

            self.slider.set_val([self.data['Время'].min() + 10, self.data['Время'].max() - 10])

            self.create_button()
            self.slider.on_changed(self.update_slider)

        def update_slider(self, val):
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the position of the vertical lines
            self.lower_limit_line.set_xdata([val[0], val[0]])
            self.upper_limit_line.set_xdata([val[1], val[1]])
            # Redraw the figure to ensure it updates
            self.fig.canvas.draw_idle()

        def create_button(self):
            button_ax = self.fig.add_axes([0.75, 0.1, 0.2, 0.05])
            self.button = wdgt.Button(ax=button_ax,
                                      label='button',
                                      color='grey',
                                      hovercolor='green'
                                      )

        def get_result(self):
            ll, ul = self.callback.get_limits()
            return ll, ul

        class RangeCallback:
            upper_limit, lower_limit = 0, 0

            def __init__(self, ax, lower_limit_line, upper_limit_line):
                self.ax = ax
                self.lower_limit_line = lower_limit_line
                self.upper_limit_line = upper_limit_line

            def click_ok(self, event):
                self.upper_limit = self.upper_limit_line.get_data()[0][0]
                self.lower_limit = self.lower_limit_line.get_data()[0][0]
                self.ax.set_title('The threshold has been successfully set!', color='green')

            def get_limits(self):
                return self.lower_limit, self.upper_limit

    def set_railway_params(self):
        picet, slpe, df_len = NovatelParser.slopes_in_main(self.data, self.profile)
        self.data = self.data.copy().iloc[len(self.data) - df_len:]
        self.data['Пикет'] = picet
        self.data['Уклон'] = slpe

    def cut_on_limits(self):
        self.data = self.data[(self.data['Время'] >= self.lower_limit) &
                              (self.data['Время'] < self.upper_limit)]

    def plot_cut(self):
        time = np.arange(0, self.data.shape[0] * 0.2, 0.2)
        if time.shape[0] > self.data.shape[0]:
            lim = abs(time.shape[0] - self.data.shape[0])
            time = time[:-lim]
        self.data['Время'] = time
        plt.plot(self.data['Время'], self.data['Скорость'])
        plt.xlabel('Время, с')
        plt.ylabel('Cкорость, м/с')
        plt.title('Зависимость скорости от времени')
        plt.show()

    def get_data_on_measured_railway(self):
        self.data = self.data[(abs(self.data['Пикет'] % 50) >= 45) | (abs(self.data['Пикет'] % 50) <= 5)]

    def calc_n_drop(self):
        self.data['pic_diff'] = self.data['Пикет'].diff()
        self.data.dropna(inplace=True)
        self.data['pic_cum'] = self.data['pic_diff'].cumsum()
        self.data.dropna(inplace=True)
        self.data = self.data.drop(columns=['Тег_данных',
                                'Корректность_данных',
                                'Широта',
                                'Долгота',
                                'Ориентация_широты',
                                'Ориентация_долготы',
                                'Дата',
                                'Уклон'])

    def heights_and_params(self):
        crop_list = []
        height_list = []
        for i, pic_exp in enumerate(self.data['Пикет']):
            for pic_prof, height_prof in self.profile[['Пикет', 'Высота']].to_numpy():
                pic_diff = abs(pic_prof - pic_exp)
                if pic_diff < 3:
                    crop_list.append(i)
                    height_list.append(height_prof)

        self.data = self.data.iloc[crop_list]
        self.data['Высота'] = height_list

    def ptr_comparison(self):
        if self.weight_state == 1:
            self.data['Wko_ptr'] = self.data['Скорость'].apply(self.ptr_gr)
        elif self.weight_state == 2:
            self.data['Wko_ptr'] = self.data['Скорость'].apply(self.ptr_por)

    def calculate_values(self):
        self.data['Расстояние'] = self.data['Пикет'].diff()
        self.data = self.data[self.data['Расстояние'] > 10]
        self.data.dropna(inplace=True)

        self.data['кв_скорости'] = self.data['Скорость'] ** 2
        self.data['delta_v_qv'] = self.data['кв_скорости'].diff()

        self.data['delta_h'] = self.data['Высота'].diff()
        self.data.dropna(inplace=True)
        self.data['Ускорение'] = ((self.data['delta_v_qv'] / 2) +
                                  (9.81 * self.data['delta_h'] / 1.06)) / self.data['Расстояние']
        time = np.arange(0, self.data.shape[0] * 0.2, 0.2)
        if time.shape[0] > self.data.shape[0]:
            lim = abs(time.shape[0] - self.data.shape[0])
            time = time[:-lim]
        self.data['Время'] = time

    def kalman_filter(self):
        dt = 0.2  # Шаг времени
        measurementSigma = 10  # Среднеквадратичное отклонение датчика
        processNoise = 1e-4  # Погрешность модели

        # Создаём объект KalmanFilter (Размер вектора стостояния, размер вектора измерений)
        filter = filterpy.kalman.KalmanFilter(dim_x=2, dim_z=1)

        # F - матрица процесса - размер dim_x на dim_x - 3х3
        filter.F = np.array([[1, dt],
                             [0, 1]])

        # Матрица наблюдения - dim_z на dim_x - 1x3
        filter.H = np.array([[1.0, 0.0]])

        # Ковариационная матрица ошибки модели
        filter.Q = filterpy.common.Q_discrete_white_noise(dim=2, dt=dt, var=processNoise)

        # Ковариационная матрица ошибки измерения - 1х1
        filter.R = np.array([[measurementSigma * measurementSigma]])

        # Начальное состояние.
        filter.x = np.array([self.kalman_init, 0.0])

        # Ковариационная матрица для начального состояния
        filter.P = np.array([[10.0, 0.0],
                             [0.0, 10.0]])

        filteredState = []
        stateCovarianceHistory = []

        # Обработка данных
        for i in range(0, self.data.shape[0]):
            self.data = self.data.astype('float64')
            z = [list(self.data['Ускорение'])[i]]  # Вектор измерений
            filter.predict()  # Этап предсказания
            filter.update(z)  # Этап коррекции

            filteredState.append(filter.x)
            stateCovarianceHistory.append(filter.P)

        filteredState = np.array(filteredState)
        stateCovarianceHistory = np.array(stateCovarianceHistory)

        # Визуализация
        plt.title("Kalman filter (2nd order)")
        plt.plot(self.data['Скорость'], self.data['Ускорение'], label="Измерение", color="#99AAFF")
        plt.plot(self.data['Скорость'], filteredState[:, 0], label="Оценка фильтра", color="#224411")
        plt.legend()
        plt.show()

        self.data['Несглаженное ускорение'] = self.data['Ускорение']
        self.data['Ускорение'] = filteredState[:, 0]

    def finishing_touches(self):
        self.weight = float(self.weight)
        self.data['Wko'] = - 1.06 * 1000 * self.data['Ускорение'] * self.weight
        self.data['Скорость'] = self.data['Скорость'] * 3.6

        if self.weight_state == 1:
            self.data['othcet'] = self.data['Скорость'].apply(self.ur_gruzh)
        elif self.weight_state == 2:
            self.data['othcet'] = self.data['Скорость'].apply(self.ur_por)

        self.data.dropna(inplace=True)

    def save_results(self):
        write_name = self.file_name.get().split('.')[-2] + '_results' + '.xlsx'
        print(write_name)
        self.data.to_excel(write_name)


def run_handler(profile, file, file_name, kalman_init, weight, weight_state):
    handler_obj = Handler(profile, file, file_name, kalman_init, weight, weight_state)
    return handler_obj.run()


if __name__ == '__main__':
    master = Tk()
    App(master)
    master.mainloop()
