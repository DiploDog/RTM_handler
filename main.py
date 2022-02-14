import os
import time
import datetime as dt
import pandas as pd
import numpy as np
import json


class NovatelParser:

    timezone = 3
    informative = True
    rus = True

    def __init__(self, file):
        self.file = file

    def __get_data(self, tag, header):
        _data = []
        with open(self.file, mode='r') as f:
            for line in f:
                if line.startswith(tag):
                    _data.append(line.strip().split(','))
        _data_arr = np.array(_data)
        _data.clear()
        return self.get_table(_data_arr, header)

    def get_gprmc(self):
        return self.__get_data('$GPRMC', 'gprmc')

    def get_table(self, array2d, data_header=None):
        with open('data' + os.sep + 'headers.json', encoding='utf-8') as h:
            headers = json.load(h)
            print(headers)

        if data_header == 'gprmc':
            df = pd.DataFrame(array2d, columns=headers['gprmc'])
            utc_time = self.parse_time(df['utc']).apply(self.to_time)
            df['utc'] = (utc_time - utc_time[0]).dt.total_seconds()
            df['date'] = self.parse_date(df['date'])
            df['Speed'] = self.knots_to_mps(df['Speed'])

            if not self.rus:
                df.rename({'Speed': 'Speed, mps',
                           'utc': 'time, s'}, axis=1, inplace=True)
                if self.informative:
                    df.drop(['track_true', 'vat_dir', 'mode_ind', '*xx'], inplace=True, axis=1)

            if self.rus:
                trans_dict = {tag[0]: tag[1] for tag in zip(headers['gprmc'], headers['gprmc_rus'])}
                df.rename(trans_dict, axis=1, inplace=True)
                if self.informative:
                    df.drop(['Отклонение от курса',
                             'Магнитное отклонение',
                             'Индикатор системы позиционирования',
                             '*xx'], inplace=True, axis=1)

            return df

        # TODO: create other headers

    @staticmethod
    def to_time(time_str):
        return dt.datetime.strptime(time_str, '%H:%M:%S:%f')

    @staticmethod
    def parse_date(column):
        return column.astype('datetime64').dt.strftime('%d-%m-%Y')

    @classmethod
    def parse_time(cls, column):
        column = pd.to_datetime(column, unit='s')
        plus_timezone = dt.timedelta(hours=cls.timezone)
        return (column + plus_timezone).dt.strftime('%H:%M:%S:%f')

    @staticmethod
    def knots_to_mps(column):
        return (column.values.astype('float32') * 0.514444).round(2)


file = NovatelParser('data/0063.data')
arr = file.get_gprmc()
arr.to_excel('test_arr_63.xlsx')
print(arr)
