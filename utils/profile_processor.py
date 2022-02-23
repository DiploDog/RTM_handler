import pandas as pd
import numpy as np
import copy
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression
from utils.error_processing import missing_file, empty_prof_data


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
        obj = PreStartPoint(self.profile_df, self.railway_peg)
        obj.calculate()
        return pd.DataFrame(data=[[obj.get_height_pred(),
                                  obj.get_lat_pred(),
                                  obj.get_long_pred(),
                                  self.railway_peg]], columns=list(self.eng_header.keys())[:-1])

    def concat_dfs(self):
        df_trunc = self.profile_df.copy()
        df_trunc = df_trunc.iloc[1:, :]
        new_df = pd.concat([df_trunc, self.new_row()], axis=0).set_index(np.arange(len(self.profile_df)))
        return new_df

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
        self.height_score = None

    def __getattr__(self, item):
        self.__dict__[item] = 0
        return 0

    def calculate(self):
        self.coords_score = self.predict_coords()
        self.height_score = self.predict_height()

    def predict_coords(self):
        X = self.prof_df['Пикет'].to_numpy().reshape(-1, 1)
        y = self.prof_df[['Широта', 'Долгота']]
        reg = LinearRegression().fit(X, y)
        score = reg.score(X, y)
        self.lat_pred, self.long_pred = reg.predict(np.array([[self.railway_peg]])).round(5)[0]
        lat_last, long_last = y.iloc[-1, :].values
        self.distance_pred = round(geodesic((self.lat_pred, self.long_pred), (lat_last, long_last)).m, 3)

        return score

    def predict_height(self):
        X = self.prof_df[['Широта', 'Долгота', 'Пикет']].to_numpy()
        y = self.prof_df['Высота']
        reg = LinearRegression().fit(X, y)
        score = reg.score(X, y)
        self.height_pred = reg.predict(np.array([[self.lat_pred, self.long_pred, self.railway_peg]])).round(2)[0]

        return score

    def get_lat_pred(self):
        return self.lat_pred

    def get_long_pred(self):
        return self.long_pred

    def get_pred_dist(self):
        return self.distance_pred

    def get_coords_score(self):
        return self.coords_score

    def get_height_score(self):
        return self.height_score

    def get_height_pred(self):
        return self.height_pred




