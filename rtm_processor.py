import cv2
from PIL import Image, ImageTk
import os
import pathlib
import tkinter
from tkinter import (StringVar, IntVar, RIDGE, Canvas, Label, Entry, Button, Radiobutton, Tk, ttk, Menu, Toplevel)
from tkinter.messagebox import showinfo, showerror, showwarning
from tkinter.filedialog import askopenfilename
from utils.profile_processor import ProfileReader
import numpy as np
import matplotlib.widgets as wdgt
import matplotlib.pyplot as plt
from novatel_parser import NovatelParser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import filterpy
from filterpy import common, kalman


def generate_degrees(source_data, degree):
    """
    Fitting polynom degree generator.

    Parameters
    ----------
    source_data : np.array, iterable
        data features
    degree : int
        degree of fitting polynom

    Returns
    -------
    np.array
        Array, that contains initial and modified source_data of initial shape according to set degree.
    """

    return np.array([
        source_data ** n for n in range(1, degree + 1)
    ]).T


def train_polynomial(x, y, xlabel='x', ylabel='y', degree=2, plot=True):
    """
    Fitting and predicting function.

    Parameters
    ----------
    x : np.array, iterable
        data features
    y : np.array, iterable
        data answers
    xlabel, ylabel : string, default: 'x', 'y'
        x-, y-axis labels
    degree : int, default: 2
        degree of fitting polynom
    plot : bool, default: True
        if True: shows the graphs

    Raises
    ------
    KeyError
        If there is no key in existing data.

    Returns
    -------
    model : object
        A LinearRegression object
    y_pred : np.array
        Predicted values to the passed data
    error : float
        Mean squared error
    coeffs : array of floats
        Polynom's coefficients
    intercept : float
        independent term of the model
    """

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
                ax.plot3D(x.iloc[:, 0], x.iloc[:, 1], y_pred)
                plt.show()

        return model, y_pred, error, coeffs, intercept


class App:
    """
    A class of the main application.

    ...

    Parameters
    ----------
    self.version : str
        String that defines the version of this program
    self.img : str
        Path to the main menu image

    Attributes
    ----------
    root : object
        Application class object
    menu : tkinter.Menu
        Application main menu
    props_menu : tkinter.Menu
        Daughter variable of menu for additional menu for Kalman filter properties
    rp_default : str
        Path to the railway profile
    railway_profile_var : tkinter.StringVar
        Text field for a path to the railway profile file
    file_to_process : tkinter.StringVar
        Text field for a path to the file with experimental data
    kalman_var : tkinter.StringVar
        Text field for initial value (float) of Kalman filter's initial condition matrix
    weight_state : tkinter.IntVar
        State of railway car: 1 - freight, 2 - empty
    weight_var : tkinter.StringVar
        The railway car weight value in tons
    weight_label : NonType
        Window label for weight_var text field representation. Type defined in weight_popup method: tkinter.Label
    kalman_props : tuple of floats
        Kalman filter properties (time step, measurement standard deviation, model error)
    time_step_var : tkinter.StringVar
        First Kalman filter property
    measurement_std_var : tkinter.StringVar
        Second Kalman filter property
    model_err_var : tkinter.StringVar
        Third Kalman filter property
    frame : tkinter.ttk.Frame
        Rectangular container for application widgets
    canvas : tkinter.Canvas
        Rectangular object for window on which other objects take their places
    kalman_props_window: tkinter.Toplevel
        Window for Kalman filter properties entering
    image_to_display : PIL.ImageTk.PhotoImage
        pillow object. Image ready to display on canvas
    profile, file, kalman_init, weight, w_s : str, str, float, float, int
        Transit values of tkitner Var objects .get() results for
        railway_profile_var, file_to_process, kalman_var, weight_var, weight_state respectively

    Methods
    -------
    init_menu():
        Start menu initialization method.
    kalman_props_menu():
        Kalman filter properties menu initialization function.
    get_kalman_props():
        The Kalman properties getter from respective menu window.
    weight_popup():
        A popup text field maker for entering the railway car weight.
    process_openfile():
        Openfile function for experimental data file.
    profile_openfile():
        Openfile function for railway profile data file.
    display_img():
        Function for displaying the start menu picture.
    raise_error_if_none(value : Any):
        static method : ValueError raiser for the cases when the passed value equals to ''
    wrong_type_error(value: Any, v_type: Type):
        static method : TypeError raiser for the cases when the passed value type does not equal to passed v_type
    get_input():
        Main menu window input getter.
    submit_insert():
        Input values setter as the class attributes nonzero values.
    go_to_process():
        A function for staring another class that is needed for calculation demanded values.
    """

    version = 'v.0.2'
    img = 'logos/main_menu_pic.png'

    def __init__(self, root):
        """
        Initializing application menu constructing.

        Parameters
        ----------
        root : object
        """

        self.root = root
        self.init_menu()

    def init_menu(self):
        """
        Start menu initialization method.
        """
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)
        self.props_menu = Menu(self.menu)
        self.menu.add_cascade(label='Properties', menu=self.props_menu)
        self.props_menu.add_command(label='Filter props', command=self.kalman_props_menu)

        self.root.title(f"Railway cars' resistance to movement handler ({self.version})")

        self.rp_default = 'data/best_profileBM.txt'
        self.railway_profile_var = StringVar()
        self.file_to_process = StringVar()
        self.kalman_var = StringVar()
        self.weight_state = IntVar()
        self.weight_var = StringVar()
        self.weight_label = None
        self.kalman_props = None

        self.time_step_var = StringVar()
        self.measurement_std_var = StringVar()
        self.model_err_var = StringVar()

        # make the frame for the canvas, on which the logos placed
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
        kalman_init.insert(0, '-0.0')

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

        self.submit_button = Button(self.root, text='Submit', command=self.submit_insert)


        # TODO: использовать widget.place() для размещения виджетов и лейблов
        railway_profile_label.pack()
        rp_of_button.pack()
        railway_profile_value.pack()
        experiment_label.pack()
        exp_of_button.pack()
        experiment_value.pack()
        kalman_init_label.pack()
        kalman_init.pack()
        rb1.pack(), rb2.pack()

    def kalman_props_menu(self):
        """
        Kalman filter properties menu initialization function.
        """

        self.kalman_props_window = Toplevel(self.root)
        self.kalman_props_window.title('Kalman filter properties')
        self.kalman_props_window.geometry('400x170')

        time_step_label = Label(self.kalman_props_window, text='Insert a model time step: ')
        time_step_entry = Entry(self.kalman_props_window, textvariable=self.time_step_var, width=60)
        if time_step_entry.get() == '':
            time_step_entry.insert(0, str(0.2))

        measurement_std_label = Label(self.kalman_props_window, text='Insert a measurement standard deviation: ')
        measurement_std_entry = Entry(self.kalman_props_window, textvariable=self.measurement_std_var, width=60)
        if measurement_std_entry.get() == '':
            measurement_std_entry.insert(0, str(5))

        model_err_label = Label(self.kalman_props_window, text='Insert a model error: ')
        model_err_entry = Entry(self.kalman_props_window, textvariable=self.model_err_var, width=60)
        if model_err_entry.get() == '':
            model_err_entry.insert(0, str(0.01))

        ok_button = Button(self.kalman_props_window, text='Ok', command=self.get_kalman_props)

        time_step_label.pack(), time_step_entry.pack()
        measurement_std_label.pack(), measurement_std_entry.pack()
        model_err_label.pack(), model_err_entry.pack()
        ok_button.pack()

    def get_kalman_props(self):
        """
        The Kalman properties getter from respective menu window.
        """

        try:
            time_step = self.time_step_var.get()
            measurement = self.measurement_std_var.get()
            model_err = self.model_err_var.get()
            time_step, measurement, model_err = float(time_step), float(measurement), float(model_err)
        except (ValueError, AttributeError, TypeError):
            showerror('Wrong Kalman properties', 'Please, enter correct Kalman properties at\n'
                                                 'Properties -> Filter props.')
        else:
            self.kalman_props = time_step, measurement, model_err
            if self.kalman_props is not None:
                showinfo('Success!', 'Kalman filter properties has been successfully set!')
                self.kalman_props_window.destroy()

    def weight_popup(self):
        """
        A popup text field maker for entering the railway car weight.
        """

        if self.weight_label is None:
            self.weight_label = Label(self.root, text='Enter car weight, tons:')
            weight_entry = Entry(self.root, textvariable=self.weight_var, width=20)
            self.weight_label.pack(), weight_entry.pack()
            self.submit_button.pack()
        else:
            pass

    def process_openfile(self):
        """
        Openfile function for experimental data file.
        """

        # curpath = pathlib.Path().resolve()
        filepath = askopenfilename(
            title='Select a file to process',
            #initialdir=curpath,
            filetypes=[
                ('.data files', '.data')
            ]
        )
        self.file_to_process.set(filepath)

    def profile_openfile(self):
        """
        Openfile function for railway profile data file.
        """

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
        """
        Function for displaying the start menu picture.
        """

        self.canvas.delete('all')
        try:
            cwd = os.getcwd()
            image_folder = os.path.join(cwd, 'logos')
            if not os.path.isdir(image_folder):
                os.mkdir(image_folder)
            image_read = cv2.imread(self.img)
            main_image = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB)
            h, w, _ = main_image.shape
            self.canvas.config(width=w, height=h)
            self.image_to_display = ImageTk.PhotoImage(Image.fromarray(main_image))
            self.canvas.create_image(w / 2, h / 2, image=self.image_to_display)
        except (FileNotFoundError, AttributeError, cv2.error):
            noimage_exception_label = Label(self.root, text='ImageLoader Error: No main menu image was found')
            noimage_exception_label.pack()

    @staticmethod
    def raise_error_if_none(value):
        """
        ValueError raiser for the cases when the passed value equals to ''

        Parameters
        ----------
        value : Any
            A value that needs to be checked on the equality to ''

        Raises
        -------
        ValueError
            if value is equal to ''
        """

        if value == '':
            raise ValueError

    @staticmethod
    def wrong_type_error(value, v_type):
        """
        TypeError raiser for the cases when the passed value type does not equal to passed v_type

        Parameters
        ----------
        value : Any
            A value which type needs to be checked on v_type type equality
        v_type : Type
            Demanded value type

        Raises
        -------
        TypeError
            If there is discrepancy between value type and v_type
        """

        if not isinstance(value, v_type):
            raise TypeError

    def get_input(self):
        """
        Main menu window input getter.

        Returns
        -------
        rp_in: str
            Railway profile input value
        proc_file_in: str
            Experimental data file input value
        kalman_in: float
            Initial (input) value of the Kalman filter's initial condition matrix
        weight_in: float
            Railway car weight input value
        weight_state_in: int
            State of railway car input value: 1 - freight, 2 - empty
        self.kalman_props: tuple of floats
            Kalman filter properties (time step, measurement standard deviation, model error) input values
        """

        try:
            rp_in = self.railway_profile_var.get()
            self.wrong_type_error(rp_in, str)
            self.raise_error_if_none(rp_in)
        except TypeError:
            showerror('Wrong value', 'Please, enter correct file path!')
        except ValueError:
            showwarning('No railway data', 'Please, enter a path for a railway data!')

        try:
            proc_file_in = self.file_to_process.get()
            self.wrong_type_error(proc_file_in, str)
            self.raise_error_if_none(proc_file_in)
        except TypeError:
            showerror('Wrong value', 'Please, enter correct file path!')
        except ValueError:
            showwarning('No experiment data', 'Please, enter a path for experiment data!')

        try:
            kalman_in = self.kalman_var.get()
            try:
                kalman_in = float(kalman_in)
            except TypeError:
                showerror('Wrong value', 'Please, enter correct value!\n'
                                         'For instance: -0.06')
            self.raise_error_if_none(kalman_in)
        except ValueError:
            showwarning('No value has been given', 'Please, enter correct value!\n'
                                                   'For instance: -0.06')

        weight_state_in = self.weight_state.get()

        try:
            weight_in = self.weight_var.get()
            try:
                weight_in = float(weight_in)
            except TypeError:
                showerror('Wrong value', 'Please, enter correct value!\n'
                                         'For instance: 100')
            self.raise_error_if_none(weight_in)
        except ValueError:
            showwarning('No value has been given', 'Please, enter correct value!\n'
                                                   'For instance: 100')

        else:
            return rp_in, proc_file_in, kalman_in, weight_in, weight_state_in, self.kalman_props

    def submit_insert(self):
        """
        Input values setter as the class attributes nonzero values.
        """

        self.profile, self.file, self.kalman_init, self.weight, self.w_s, self.kalman_props = self.get_input()
        if self.kalman_props is None:
            self.kalman_props = [0.2, 10, 1e-2]
        try:
            if any(np.array(
                    [self.profile, self.file, self.kalman_init,
                     self.weight, self.w_s, self.kalman_props[0],
                     self.kalman_props[1], self.kalman_props[2]]) == ''):
                showerror('Not enough data to proceed', 'Please, fill in all the text fields!')
            else:
                self.process_button = Button(self.root, text='Process', command=self.go_to_process)
                self.process_button.pack()
        except TypeError:
            showerror('Not enough data to proceed', 'Please, fulfill all the text fields')

    def go_to_process(self):
        """
        A function for staring another class that is needed for calculation demanded values.
        """

        self.root.quit()
        run_handler(
            self.profile, self.file, self.file_to_process,
            self.kalman_init, self.weight, self.w_s, self.kalman_props
        )
        self.process_button.destroy()


class Handler:
    """
    Handles all processing calculations and holds the MPL GUI class.

    Attributes
    ----------
    profile : str
        Path to the railway profile
    file : str
        Path to the experimental data file
    file_name : tkinter.StringVar
        Name of the file to save (part of the file path)
    kalman_init : float
        Initial value of the Kalman filter's initial condition matrix
    weight: float
        Railway car weight value
    weight_state: int
        State of railway car input value: 1 - freight, 2 - empty
    kalman_props: tuple of floats
        Kalman filter properties (time step, measurement standard deviation, model error) input values
    data : pd.DataFrame
        Modified dataframe of the experimental and calculated data
    lower_limit : float
        Lower time limit of the experiment (chose by user)
    upper_limit : float
        Upper time limit of the experiment (chose by user)

    Methods
    -------
    run():
        A method that runs all the other methods sequentially.
    define_profile():
        Uses ProfileReader class, that performs the preparations
        and mathing with experimental data and sets the profile attribute as pd.DataFrame.
    set_data():
        Uses NovatelParser class that performs the GPS navigator data parsing
        which returns the dataframe and sets the data class attribute as pd.DataFrame.
    linearize_profile():
        Sets the railway peg value due to the measured latitude and longitude
        for the data dataframe.
    ur_gruzh(v: float, pd.Series, array): -> float, pd.Series, array
        Calculates the main resistance to motion (N/t) of the laden gondola car (12-196-01)
        with the formula obtained with the VNIIZHT experiments and computer models calculations.
    ur_por(v: float, pd.Series, array): -> float, pd.Series, array
        Calculates the main resistance to motion (N/t) of the empty gondola car (12-196-01)
        with the formula obtained with the VNIIZHT experiments and computer models calculations.
    ptr_gr(v: float, pd.Series, array): -> float, pd.Series, array
        Calculates the main resistance to motion (N/t) of the laden gondola car (12-196-01)
        according to the Traction calculation rules for train operation
    ptr_por(v: float, pd.Series, array): -> float, pd.Series, array
        Calculates the main resistance to motion (N/t) of the empty gondola car (12-196-01)
        according to the Traction calculation rules for train operation.
    set_limits():
        Runs the additional GU interface for selecting and setting
        the useful experimental time limits with the MplGUI class.
    set_railway_params():
        Railway peg and slope sections setter.

        Cuts the dataframe according to the obtained time limits.

    Subclasses
    ----------
    MplGUI
        The additional GU interface for selecting and setting the useful experimental time limits.

        RangeCallback
            GUI callback class for submitting and defining
            the positions of limit lines
    """

    def __init__(self, profile, file, file_name, kalman_init, weight, weight_state, kalman_props):
        self.profile = profile
        self.file = file
        self.file_name = file_name
        self.kalman_init = kalman_init
        self.weight = weight
        self.weight_state = weight_state
        self.kalman_props = kalman_props
        self.data = None
        self.lower_limit = None
        self.upper_limit = None

    def run(self):
        """
        A method that runs all the other methods sequentially.
        """

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
        """
        Uses ProfileReader class, that performs the preparations
        and mathing with experimental data and sets the profile attribute as pd.DataFrame.
        """

        self.profile = ProfileReader(self.profile).set_profile()

    def set_data(self):
        """
        Uses NovatelParser class that performs the GPS navigator data parsing
        which returns the dataframe and sets the data class attribute as pd.DataFrame.
        """

        self.data = NovatelParser(self.file, 0).get_gprmc(fltr=False)

    def linearize_profile(self):
        """
        Sets the railway peg value due to the measured latitude and longitude
        for the data dataframe.
        """

        lr = LinearRegression()
        x = self.profile[['Широта', 'Долгота']]
        y = self.profile['Пикет']
        lr.fit(x, y)
        self.data['Пикет'] = lr.predict(self.data[['Широта', 'Долгота']] * -1)

    @staticmethod
    def ur_gruzh(v):
        """
        Calculates the main resistance to motion (N/t) of the laden gondola car (12-196-01)
        with the formula obtained with the VNIIZHT experiments and computer models calculations.

        Parameters
        ----------
        v : float, pd.Series, array
            Speed of the railway car

        Returns
        -------
        float, pd.Series, array
            The main resistance to motion (N/t) of the laden gondola car (12-196-01)
        """

        return 0.4696 * v ** 2 - 0.2287 * v + 488.13

    @staticmethod
    def ur_por(v):
        """
        Calculates the main resistance to motion (N/t) of the empty gondola car (12-196-01)
        with the formula obtained with the VNIIZHT experiments and computer models calculations.

        Parameters
        ----------
        v : float, pd.Series, array
            Speed of the railway car

        Returns
        -------
        float, pd.Series, array
            The main resistance to motion (N/t) of the empty gondola car (12-196-01)
        """

        return 0.5441 * v ** 2 - 2.1127 * v + 244.78

    @staticmethod
    def ptr_gr(v):
        """
        Calculates the main resistance to motion (N/t) of the laden gondola car (12-196-01)
        according to the Traction calculation rules for train operation.

        Parameters
        ----------
        v : float, pd.Series, array
            Speed of the railway car

        Returns
        -------
        ptr_wko : float, pd.Series, array
            The main resistance to motion (N/t) of the laden gondola car (12-196-01)
        """

        v = v * 3.6
        ptr = 9.81 * 100 * (0.7 + (3 + 0.09 * v + 0.002 * v ** 2) / 25)
        aero_multi = 0.0611 * v ** 2 + 0.8275 * v
        aero_solo = 0.4384 * v ** 2 - 0.2071 * v
        ptr_wko = ptr - aero_multi + aero_solo

        return ptr_wko

    @staticmethod
    def ptr_por(v):
        """
        Calculates the main resistance to motion (N/t) of the empty gondola car (12-196-01)
        according to the Traction calculation rules for train operation.

        Parameters
        ----------
        v : float, pd.Series, array
            Speed of the railway car

        Returns
        -------
        ptr_wko : float, pd.Series, array
            The main resistance to motion (N/t) of the empty gondola car (12-196-01)
        """

        v = v * 3.6
        ptr = 9.81 * 25 * (1.0 + 0.044 * v + 0.00021 * v ** 2)
        aero_multi = 0.0637 * v ** 2 + 1.2434 * v
        aero_solo = 0.5218 * v ** 2 - 0.6131 * v
        ptr_wko = ptr - aero_multi + aero_solo

        return ptr_wko

    def set_limits(self):
        """
        Runs the additional GU interface for selecting and setting
        the useful experimental time limits with the MplGUI class.
        """

        mplgui_object = self.MplGUI(self.data)
        self.lower_limit, self.upper_limit = mplgui_object.get_result()

    class MplGUI:
        """
        The additional GU interface for selecting and setting the useful experimental time limits.

        Attributes
        ----------
        data : pd.DataFrame
            Modified dataframe of the experimental and calculated data
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Respectively: The top level container for all the plot elements;
                          Contains the figure elements
        lower_limit_line : NonType object
            Used to set the matplotlib.lines.Line2D in GUI as the class attribute
            for the lower time limit
        upper_limit_line : NonType object
            Used to set the matplotlib.lines.Line2D in GUI as the class attribute
            for the upper time limit
        slider : matplotlib.widgets.RangeSlider
            Interactive slider object to choose the time range
        button : matplotlib.widgets.Button
            Used for setting the slider lower and upper limits values
        callback : Handler.MplGUI.RangeCallback
            Used for getting the callback of the GUI and user interaction

        Methods
        -------
        init_plot():
            Plots the initial speed vs time graph.
        create_widgets():
            Method which creates the widgets: range slider and submit button.
        update_slider(val: tuple, list, iterable):
            Slider lines updater.
        create_button():
            Creates the submit button.
        get_result(): -> tuple of floats
            Lower and upper lines getter (interacts with callback method)
        """

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
            """
            Plots the initial speed vs time graph.
            """

            self.fig, self.ax = plt.subplots()
            plt.subplots_adjust(left=0.1, bottom=0.35)
            plt.plot(self.data['Время'], self.data['Скорость'])
            plt.xlabel('Время, с')
            plt.ylabel('Cкорость, м/с')
            plt.title('Зависимость скорости от времени')

        def create_widgets(self):
            """
            Method which creates the widgets: range slider and submit button.
            """

            slider_ax = self.fig.add_axes([0.20, 0.2, 0.60, 0.03])
            self.slider = wdgt.RangeSlider(ax=slider_ax,
                                           label="Threshold",
                                           valmin=self.data['Время'].min(),
                                           valmax=self.data['Время'].max(),
                                           valstep=0.2,
                                           valinit=[
                                               self.data['Время'].min() + 10,
                                               self.data['Время'].max() - 10
                                           ])

            self.lower_limit_line = self.ax.axvline(self.slider.val[0], color='k')
            self.upper_limit_line = self.ax.axvline(self.slider.val[1], color='k')

            self.slider.set_val([self.data['Время'].min() + 10, self.data['Время'].max() - 10])

            self.create_button()
            self.slider.on_changed(self.update_slider)

        def update_slider(self, val):
            """
            Slider lines updater.

            Parameters
            ----------
            val : tuple, list, iterable
                The x value of the graph coordinates
            """

            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the position of the vertical lines
            self.lower_limit_line.set_xdata([val[0], val[0]])
            self.upper_limit_line.set_xdata([val[1], val[1]])
            # Redraw the figure to ensure it updates
            self.fig.canvas.draw_idle()

        def create_button(self):
            """
            Creates the submit button.
            """

            button_ax = self.fig.add_axes([0.75, 0.1, 0.2, 0.05])
            self.button = wdgt.Button(ax=button_ax,
                                      label='button',
                                      color='grey',
                                      hovercolor='green'
                                      )

        def get_result(self):
            """
            Lower and upper lines getter (interacts with callback method)

            Returns
            -------
            ll : float
                Lower limit lime position (x)
            ul : float
                Upper limit lime position (x)
            """

            ll, ul = self.callback.get_limits()
            return ll, ul

        class RangeCallback:
            """
            GUI callback class for submitting and defining
            the positions of limit lines

            Parameters
            ----------
            self.upper_limit : int, float
                Initial upper limit value
            self.upper_limit : int, float
                Initial upper limit value

            Attributes
            ----------
            ax : matplotlib.axes.Axes
                Figure elements container
            lower_limit_line : matplotlib.lines.Line2D
                Lower limit line position display
            upper_limit_line : matplotlib.lines.Line2D
                Upper limit line position display

            Methods
            -------
            click_ok(event: result of user and mpl interaction):
                Set the final tuple of limits values by user if ok button clicked
            get_limits(): -> tuple of floats
                Limit values getter
            """

            upper_limit, lower_limit = 0, 0

            def __init__(self, ax, lower_limit_line, upper_limit_line):
                self.ax = ax
                self.lower_limit_line = lower_limit_line
                self.upper_limit_line = upper_limit_line

            def click_ok(self, event):
                """
                Set the final tuple of limits values by user if ok button clicked

                Parameters
                ----------
                event : result of user and mpl interaction

                """
                self.ax.set_title('The threshold has been successfully set!', color='green')
                self.upper_limit = self.upper_limit_line.get_data()[0][0]
                self.lower_limit = self.lower_limit_line.get_data()[0][0]

            def get_limits(self):
                """
                Limit values getter

                Returns
                -------
                tuple of floats:
                    Limit values
                """
                return self.lower_limit, self.upper_limit

    def set_railway_params(self):
        """
        Railway peg and slope sections setter.
        """

        picet, slpe, df_len = NovatelParser.slopes_in_main(self.data, self.profile)
        self.data = self.data.copy().iloc[len(self.data) - df_len:]
        self.data['Пикет'] = picet
        self.data['Уклон'] = slpe

    def cut_on_limits(self):
        """
        Cuts the dataframe according to the obtained time limits.
        """

        self.data = self.data[(self.data['Время'] >= self.lower_limit) &
                              (self.data['Время'] < self.upper_limit)]

    def plot_cut(self):
        """
        Proper dataframe cutter with the time fitter
        Also plots the informative approximation of speed vs time 2th polynom
        """

        time = np.arange(0, self.data.shape[0] * 0.2, 0.2)
        if time.shape[0] > self.data.shape[0]:
            lim = abs(time.shape[0] - self.data.shape[0])
            time = time[:-lim]
        self.data['Время'] = time
        train_polynomial(self.data['Время'],
                         self.data['Скорость'],
                         'Время', 'Скорость', 2, True)

    def get_data_on_measured_railway(self):
        """
        Cuts the data by only true railway peg or half-peg values
        """

        self.data = self.data[(abs(self.data['Пикет'] % 50) >= 45) | (abs(self.data['Пикет'] % 50) <= 5)]

    def calc_n_drop(self):
        """
        Calculates the differences between passed distance in sequential rows of dataframe,
        calculates the cumulative sum of sequentially passed distance pieces by rows,
        drops the useless for future processing columns.
        """

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
        """
        Matching heights with the position of the railway
        """

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
        """
        Adds the main resistance value according to
        the Traction calculation rules for train operation and the car weight state:
        Laden if weight_state = 1,
        Empty if weight_state = 2.
        """

        if self.weight_state == 1:
            self.data['Wko_ptr'] = self.data['Скорость'].apply(self.ptr_gr)
        elif self.weight_state == 2:
            self.data['Wko_ptr'] = self.data['Скорость'].apply(self.ptr_por)

    def calculate_values(self):
        """
        Calculates the auxiliary values
        """

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
        """
        Method for Kalman's filter implementation
        """

        # time step, sensor's standard deviation, model error
        dt, measurementSigma, processNoise = self.kalman_props

        # creating the KalmanFilter object (Shape of state vector, Shape of measure vector)
        filter = filterpy.kalman.KalmanFilter(dim_x=2, dim_z=1)

        # F - process matrix with the shape of (dim_x, dim_x)
        filter.F = np.array([[1, dt],
                             [0, 1]])

        # Observation matrix with the shape of (dim_z, dim_x)
        filter.H = np.array([[1.0, 0.0]])

        # Covariance matrix of model's error
        filter.Q = filterpy.common.Q_discrete_white_noise(dim=2, dt=dt, var=processNoise)

        # Covariance matrix of measuring error (1, 1)
        filter.R = np.array([[measurementSigma * measurementSigma]])

        # Initial condition
        filter.x = np.array([self.kalman_init, 0.0])

        # Covariance matrix of initial condition
        filter.P = np.array([[10.0, 0.0],
                             [0.0, 10.0]])

        filteredState = []
        stateCovarianceHistory = []

        # data processing
        for i in range(0, self.data.shape[0]):
            self.data = self.data.astype('float64')
            z = [list(self.data['Ускорение'])[i]]  # Measuring vector
            filter.predict()  # Prediction
            filter.update(z)  # Correction

            filteredState.append(filter.x)
            stateCovarianceHistory.append(filter.P)

        filteredState = np.array(filteredState)
        stateCovarianceHistory = np.array(stateCovarianceHistory)

        # Visualization
        plt.title("Kalman filter (2nd order)")
        plt.plot(self.data['Скорость'], self.data['Ускорение'], label="Измерение", color="#99AAFF")
        plt.plot(self.data['Скорость'], filteredState[:, 0], label="Оценка фильтра", color="#224411")
        plt.legend()
        plt.show()

        self.data['Несглаженное ускорение'] = self.data['Ускорение']
        self.data['Ускорение'] = filteredState[:, 0]

    @staticmethod
    def construct_equation(v, a, b):
        return a * v**2 + b * v

    def finishing_touches(self):
        """
        Calculates and adds the column with the main resistance to motion of the railway car
        due to the experimental data
        Adds the main resistance value according to the formula obtained with
        the VNIIZHT experiments and computer models calculations and the car weight state:
        Laden if weight_state = 1,
        Empty if weight_state = 2.
        """

        self.weight = float(self.weight)
        self.data['Wko'] = - 1.06 * 1000 * self.data['Ускорение'] * self.weight

        self.data['Скорость'] = self.data['Скорость'] * 3.6

        if self.weight_state == 1:
            self.data['othcet'] = self.data['Скорость'].apply(self.ur_gruzh)
            self.data['Wko_v_sostave'] = self.data['Wko'] - \
                                         self.data['Скорость'].apply(
                                             lambda x: self.construct_equation(x, 0.4384, -0.2071)) + \
                                         self.data['Скорость'].apply(
                                             lambda x: self.construct_equation(x, 0.0611, 0.8275))
        elif self.weight_state == 2:
            self.data['othcet'] = self.data['Скорость'].apply(self.ur_por)
            self.data['Wko_v_sostave'] = self.data['Wko'] - \
                                         self.data['Скорость'].apply(
                                             lambda x: self.construct_equation(x, 0.5218, -0.6131)) + \
                                         self.data['Скорость'].apply(
                                             lambda x: self.construct_equation(x, 0.0637, 1.2434))

        self.data.dropna(inplace=True)

    def save_results(self):
        """
        Saves the calculation results in excel file for further processing if needed
        """

        write_name = self.file_name.get().split('.')[-2] + '_results' + '.xlsx'
        wn_for_user = '~/' + '/'.join(write_name.split('/')[-3:])
        self.data.to_excel(write_name)
        showinfo(
            f'Processing finished',
            f'The processed data has been saved to\n'
            f'{wn_for_user}'
        )


def run_handler(profile, file, file_name, kalman_init, weight, weight_state, kalman_props):
    """
    Runs the data processing with all the GUIs

    Parameters
    ----------
    profile : str
        Path to the railway profile
    file : str
        Path to the experimental data file
    file_name : tkinter.StringVar
        Name of the file to save (part of the file path)
    kalman_init : float
        Initial value of the Kalman filter's initial condition matrix
    weight : float
        Railway car weight value
    weight_state : int
        State of railway car input value: 1 - freight, 2 - empty
    kalman_props : tuple of floats
        Kalman filter properties (time step, measurement standard deviation, model error) input values

    Returns
    -------
    None
        Runs the processing method of Handler class
    """

    handler_obj = Handler(profile, file, file_name, kalman_init, weight, weight_state, kalman_props)
    return handler_obj.run()


if __name__ == '__main__':
    master = Tk()
    App(master)
    master.mainloop()
