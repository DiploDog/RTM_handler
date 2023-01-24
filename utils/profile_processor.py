import pandas as pd
import numpy as np
from copy import copy
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression


class ProfileReader:
    """
    Railway profile reader class.
    Reads and prepare raw profile data for uses in experimental data processing

    Parameters
    ----------
    self.eng_header : dict
        Eng-rus translation of the file columns
    self.rus_header : dict
        Rus-eng translation of the file columns

    Attributes
    ----------
    profile : str
        A path to the railway profile file
    railway_peg : float
        The last railway peg on the test track
    rus : str
        optional: 'y' if eng-rus translation is needed, 'n' if not
    temp_data : list
        Temporary data buffer
    profile_df : pd.DataFrame
        The .get_profile() method invoker
    obj_psp : NonType
        PreStartPoint class object

    Methods
    -------
    set_widen_df():
        Widen dataframe setter.
    get_profile(): -> pd.DataFrame
        Final processing method, using translation attributes.
    widen_dfs(): -> pd.DataFrame
        Widened dataframe constructor.
    calculate_slope():
        Calculates slope of the profile according to the railway rail heights.
    read_text():
        Method for reading profile.
    to_dataframe(): -> pd.DataFrame
        Makes the dataframe's all values' types float.
    new_row():
        Uses PreTartPoint class for predicting new row as the dataframe.
    shift_df(): -> pd.DataFrame
        Truncates initial dataframe by deleting first raw.
        Concatenating shifted dataframe with the initial.
    get_widened_df(): -> pd.DataFrame
        Widen dataframe getter.
    set_profile(): invokes get_widened_df()
        Prepared railway profile setter, _get_widened_df method invoker.
    to_float(val: int, str, pd.Series, array): -> float
        ToFloat data transformer.
    save_data(line: str, temp_data: list):
        Fills the temporary data buffer list.
    clear_data(temp_data: list):
        Method for deleting all elements from the temp_data and saving computer memory.
    """

    eng_header = {
        'Высота': 'Height, m',
        'Широта': 'Latitude',
        'Долгота': 'Longitude',
        'Пикет': 'Railway peg',
        'Уклон': 'Slope'
    }

    rus_header = {
        'Height': 'Высота',
        'Latitude': 'Широта',
        'Longitude': 'Долгота',
        'Railway_peg': 'Пикет',
        'Slope': 'Уклон'
    }

    def __init__(self, profile, railway_peg=9.1, rus=True):
        self.profile = profile
        self.railway_peg = railway_peg
        self.rus = 'n'
        self.temp_data = []
        self.profile_df = self.get_profile()
        self.widened_df = None
        self.obj_psp = None

    def set_widen_df(self):
        """
        Widen dataframe setter.
        """

        self.widened_df = self.widen_dfs()

    def get_profile(self):
        """
        Final processing method, using translation attributes.

        Raises
        ------
        IndexError
            If profile data is empty

        Returns
        -------
        profile_df : pd.DataFrame
            Final railway profile dataframe
        """

        self.read_txt()
        try:
            profile_df = self.to_dataframe()
            self.clear_data(self.temp_data)
            if self.rus == 'y':
                profile_df.rename(self.eng_header, axis=1, inplace=True)
            elif self.rus == 'n':
                profile_df.rename(self.rus_header, axis=1, inplace=True)
            return profile_df

        except IndexError:
            if self.rus:
                print('Файл профиля пути пуст.')
            else:
                print('Profile data is empty.')

    def widen_dfs(self):
        """
        Widened dataframe constructor.

        Returns
        -------
        widened_df : pd.DataFrame
            Widened dataframe
        """

        shifted_df = self.shift_df()
        widened_df = self.profile_df.join(shifted_df, how='left', rsuffix='_след.')
        return widened_df

    def calculate_slope(self):
        """
        Calculates slope of the profile according to the railway rail heights.

        Raises
        ------
        ValueError
            If geo data is incorrect
        """

        self.set_widen_df()
        df = self._get_widened_df()
        if df is not None:
            df['Расстояние'] = df.apply(lambda row: PreStartPoint.geodistance(
                (row['Широта_след.'], row['Долгота_след.']),
                (row['Широта'], row['Долгота'])), axis=1).round(2)
            df['Уклон'] = \
                ((df['Высота_след.'] - df['Высота']) / df['Расстояние']).round(4)
            self.widened_df = df
        else:
            try:
                raise ValueError
            except ValueError:
                if self.rus:
                    print('Невозможно рассчитать уклон. Проверьте данные массива.')
                else:
                    print('Can not calculate the slope. Check the array data.')

    def read_txt(self):
        """
        Method for reading profile.

        Raises
        ------
        FileNotFoundError
            If there is no file with the profile filename
        """

        try:
            with open(self.profile, mode='r', encoding='utf-8') as f:
                for line in f:
                    self.save_data(line, self.temp_data)

        except FileNotFoundError:
            if self.rus:
                print('Ошибка! Файл отсутствует или поврежден.')
            else:
                print('Error! File does not exist or invalid.')

    def to_dataframe(self):
        """
        Makes the dataframe's all values' types float.

        Returns
        -------
        floated_df : pd.DataFrame
            DataFrame with float values
        """

        columns = self.temp_data[0]
        data = self.temp_data[1:]

        dataframe = pd.DataFrame(data=data, columns=columns)
        floated_df = dataframe.applymap(self.to_float)

        return floated_df

    def new_row(self):
        """
        Uses PreTartPoint class for predicting new row as the dataframe.

        Returns
        -------
        pd.DataFrame
            New predicted row of the profile dataframe
        """

        self.obj_psp = PreStartPoint(self.profile_df, self.railway_peg)
        self.obj_psp.calculate()

        return pd.DataFrame(data=[[self.obj_psp.get_height_pred(),
                                  self.obj_psp.get_lat_pred(),
                                  self.obj_psp.get_long_pred(),
                                  self.railway_peg]], columns=list(self.eng_header.keys())[:-1])

    def shift_df(self):
        """
        Truncates initial dataframe by deleting first raw.
        Concatenating shifted dataframe with the initial.

        Returns
        -------
        new_df : pd.DataFrame
            Concatenated dataframes: initial and shifted
        """

        df_trunc = copy(self.profile_df)
        df_trunc = df_trunc.iloc[1:, :]
        new_df = pd.concat([df_trunc, self.new_row()], axis=0).set_index(np.arange(len(self.profile_df)))

        return new_df

    def _get_widened_df(self):
        """
        Widen dataframe getter.

        Returns
        -------
        self.widened_df: pd.DataFrame
            Widened dataframe
        """

        return self.widened_df

    def set_profile(self):
        """
        Prepared railway profile setter, _get_widened_df method invoker.

        Returns
        -------
        self._get_widened_df() method
        """

        profile = self.get_profile()
        if profile['Уклон'][0]:
            self.set_widen_df()
            df = self._get_widened_df()
            return df
        self.calculate_slope()

        return self._get_widened_df()

    @staticmethod
    def to_float(val):
        """
        ToFloat data transformer.

        Parameters
        ----------
        val : int, str, pd.Series, array

        Returns
        -------
        float
            Value(s) transformed to float value(s)
        """
        return float(val)

    @staticmethod
    def save_data(line, temp_data):
        """
        Fills the temporary data buffer list.

        Parameters
        ----------
        line : str
            The text file line read
        temp_data: list
            Temporary data buffer
        """

        str_list = line.strip().split()
        temp_data.append(str_list)

    @staticmethod
    def clear_data(temp_data):
        """
        Method for deleting all elements from the temp_data and saving computer memory.

        Parameters
        ----------
        temp_data: list
            Temporary data buffer
        """
        temp_data.clear()


class PreStartPoint:
    """
    ML class for predicting railway parameters
    Optional class

    Attributes
    ----------
    prof_df : pd.DataFrame
        Initial railway profile dataframe
    railway_peg : float
        The last railway peg on the test track
    lat_pred : NonType, float
        Predicted latitude
    long_pred : NonType, float
        Predicted longitude
    height_pred : NonType, float
        Predicted height
    distance_pred : float
        Predicted distance
    coords_score : float
        Predicting score

    Methods
    -------
    calculate():
        Predicted coordinates setter.
    predict_coords(): -> float
        Coordinates predictor.
    geodistance(point1: tuple of floats, point2: tuple of floats): -> float
        Calculates the distance between two coordinate points.
    get_lat_pred(): -> float
        Predicted latitude getter.
    get_long_pred(): -> float
        Predicted longitude getter.
    get_pred_dist(): -> float
        Predicted distance getter.
    get_coords_score(): -> float
        Prediction score for coordinates getter.
    get_height_pred(): -> float
        Predicted height getter.
    """

    def __init__(self, prof_df, railway_peg):
        self.prof_df = prof_df
        self.railway_peg = railway_peg

        self.lat_pred = None
        self.long_pred = None
        self.height_pred = None
        self.distance_pred = None
        self.coords_score = None

    def __getattr__(self, item):
        self.__dict__[item] = 0
        return 0

    def calculate(self):
        """
        Predicted coordinates setter.
        """

        self.coords_score = self.predict_coords()

    def predict_coords(self):
        """
        Coordinates predictor.

        Returns
        -------
        score: float
            Predicting score
        """
        X = self.prof_df['Пикет'].to_numpy().reshape(-1, 1)
        y = self.prof_df[['Высота', 'Широта', 'Долгота']]
        reg = LinearRegression().fit(X, y)
        score = reg.score(X, y)
        height, self.lat_pred, self.long_pred = reg.predict(np.array([[self.railway_peg]])).round(5)[0]
        _, lat_last, long_last = y.iloc[-1, :].values
        self.height_pred = round(height, 2)
        self.distance_pred = round(self.geodistance((self.lat_pred, self.long_pred), (lat_last, long_last)), 3)

        return score

    @staticmethod
    def geodistance(point1, point2):
        """
        Calculates the distance between two coordinate points.

        Parameters
        ----------
        point1 : tuple of floats
            First coordinate (latitude, longitude)
        point2 : tuple of floats
             Second coordinate (latitude, longitude)
        Returns
        -------
        Result of geodesic method, tuple of floats
        """

        return geodesic(point1, point2).m

    def get_lat_pred(self):
        """
        Predicted latitude getter.

        Returns
        -------
        self.lat_pred : float
            Prediction result for latitude
        """
        return self.lat_pred

    def get_long_pred(self):
        """
        Predicted longitude getter.

        Returns
        -------
        self.long_pred : float
            Prediction result for longitude
        """
        return self.long_pred

    def get_pred_dist(self):
        """
        Predicted distance getter.

        Returns
        -------
        self.distance_pred : float
            Prediction result for distance
        """
        return self.distance_pred

    def get_coords_score(self):
        """
        Prediction score for coordinates getter.

        Returns
        -------
        self.coords_score : float
            Prediction score result for coordinates
        """
        return self.coords_score

    def get_height_pred(self):
        """
        Predicted height getter.

        Returns
        -------
        self.height_pred : float
            Prediction result for height
        """
        return self.height_pred




