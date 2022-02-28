import pandas as pd
import numpy as np
from copy import copy
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression
from utils.error_messages import *


class ProfileReader:

    eng_header = {'Высота': 'Height, m',
                  'Широта': 'Latitude',
                  'Долгота': 'Longitude',
                  'Пикет': 'Railway peg',
                  'Уклон': 'Slope'}

    def __init__(self, profile, railway_peg=9.1, rus=True):
        self.profile = profile
        self.railway_peg = railway_peg
        self.rus = rus
        self.temp_data = []
        self.profile_df = self.get_profile()
        self.widened_df = None
        self.obj_psp = None

    def set_widen_df(self):
        self.widened_df = self.widen_dfs()

    def get_profile(self):
        self.read_txt()
        try:
            profile_df = self.to_dataframe()
            self.clear_data(self.temp_data)
            if not self.rus:
                profile_df.rename(self.eng_header, axis=1, inplace=True)
            return profile_df

        except IndexError:
            empty_prof_data(rus=self.rus)

    def widen_dfs(self):
        shifted_df = self.shift_df()
        widened_df = self.profile_df.join(shifted_df, how='left', rsuffix='_след.')
        return widened_df

    def calculate_slope(self):
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
                no_widened_data(rus=self.rus)

    def read_txt(self):
        try:
            with open(self.profile, mode='r') as f:
                for line in f:
                    self.save_data(line, self.temp_data)

        except FileNotFoundError:
            missing_file(rus=self.rus)

    def to_dataframe(self):
        columns = self.temp_data[0]
        data = self.temp_data[1:]

        dataframe = pd.DataFrame(data=data, columns=columns)
        floated_df = dataframe.applymap(self.to_float)
        return floated_df

    def new_row(self):
        self.obj_psp = PreStartPoint(self.profile_df, self.railway_peg)
        self.obj_psp.calculate()
        return pd.DataFrame(data=[[self.obj_psp.get_height_pred(),
                                  self.obj_psp.get_lat_pred(),
                                  self.obj_psp.get_long_pred(),
                                  self.railway_peg]], columns=list(self.eng_header.keys())[:-1])

    def shift_df(self):
        df_trunc = copy(self.profile_df)
        df_trunc = df_trunc.iloc[1:, :]
        new_df = pd.concat([df_trunc, self.new_row()], axis=0).set_index(np.arange(len(self.profile_df)))
        return new_df

    def _get_widened_df(self):
        return self.widened_df

    # TODO: Расчитать уклоны по profileBM.txt

    @staticmethod
    def to_float(val):
        return float(val)

    @staticmethod
    def save_data(line, temp_data):
        str_list = line.strip().split()
        temp_data.append(str_list)

    @staticmethod
    def clear_data(temp_data):
        temp_data.clear()


class PreStartPoint:

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
        self.coords_score = self.predict_coords()

    def predict_coords(self):
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
        return geodesic(point1, point2).m

    def get_lat_pred(self):
        return self.lat_pred

    def get_long_pred(self):
        return self.long_pred

    def get_pred_dist(self):
        return self.distance_pred

    def get_coords_score(self):
        return self.coords_score

    def get_height_pred(self):
        return self.height_pred




