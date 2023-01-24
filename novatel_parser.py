from utils.dms import dms, decimal_degrees
from utils.profile_processor import ProfileReader
import os
import datetime as dt
import pandas as pd
import numpy as np
import json
from tkinter.messagebox import showerror


class NovatelParser:
    """
    Parsing class. Returns the experimental data as the pd.DataFrame
    after preprocessing the Novatel GPS navigator raw data

    Parameters
    ----------
    self.timezone : int
        GMT lead or lag time in hours
    self.informative : bool
        True if additional GPS navigator informative columns are needed in report
    self.rus : bool
        True if English-Russian translation is needed

    Attributes
    ----------
    file : str
        A path to the experimental data file
    freq : int
        The data filter frequency
    fltr : bool
        True if data filtration is needed
    table : NonType
        Var for the future pd.DataFrame which would contain the processed experimental data
    profile : NonType
        A path to the railway profile file

    Methods
    -------
    __get_data(tag: str, freq: int): -> pd.DataFrame
        Parse method, collects data from .data or .txt files.
    get_gpgga(): -> pd.DataFrame
        $GPGGA tag data getter.
    get_gprmc(fltr: int): -> pd.DataFrame
        $GPRMC tag data getter.
        If "filter" has to be used (fltr = True), then lines will be sparse
        according to the fltr integer value. i.e. if fltr = 2,
        every second line will be excluded. Optional.
    set_profile(): -> pd.DataFrame
        Railway profile setter. Uses ProfileReader class.
    headers_loader():
        Loading headers for result table from the .json file.
    get_table(array2d: pd.DataFrame, data_header: None (dict)): -> pd.DataFrame
        Result table constructor.
    filt_truncate(dataframe: pd.DataFrame, filter_value: int): -> pd.DataFrame
        If filter is used, return sparse experimental dataframe.
    to_time(time_str: pd.Series): -> str, pd.Series
        To-time dataframe convertor.
    parse_date(column: pd.Series): -> pd.Series
        To-datetime dataframe convertor.
    parse_time(column: pd.Series): -> pd.Series
        UTS datetime convertor.
    knots_to_mps(column: pd.Series): -> pd.Series
        Knots to meter per seconds convertor.
    coord_processing(coord: pd.DataFrame, pd.Series, np.array, direction: str): -> float
        The decimal format of degree transformer and coordinate format reducer.
    slopes_in_main(arr: pd.DataFrame, auxil: pd.DataFrame): -> tuple(list, list, float)
        Railway slope calculator.
    """

    timezone = 3
    informative = True
    rus = True

    def __init__(self, file, freq):
        self.file = file
        self.freq = freq
        self.fltr = False
        self.table = None
        self.profile = None

    def __get_data(self, tag, header):
        """
        Parse method, collects data from .data or .txt files.

        Parameters
        ----------
        tag : str
            Novatel GPS tag on which demanded data has to be searched in the file
        header : str
            Column name according to name of the tag

        Returns
        -------
        pd.DataFrame
            Parsed table with Novatel GPS navigator
        """

        _data = []

        with open(self.file, mode='r') as f:
            for line in f:
                if line.startswith(tag):
                    _data.append(line.strip().split(','))
        _data_arr = np.array(_data)
        _data.clear()

        return self.get_table(_data_arr, header)

    def get_gpgga(self):
        """
        $GPGGA tag data getter.

        Returns
        -------
        pd.DataFrame
            Parsed table with Novatel GPS navigator
        """
        return self.__get_data('$GPGGA', 'gpgga')

    def get_gprmc(self, fltr):
        """
        $GPRMC tag data getter.
        If "filter" has to be used (fltr = True), then lines will be sparse
        according to the fltr integer value. i.e. if fltr = 2,
        every second line will be excluded. Optional.

        Parameters
        ----------
        fltr : int
            Decimation frequency

        Returns
        -------
        pd.DataFrame
            Parsed table with Novatel GPS navigator
        """

        if fltr:
            self.fltr = fltr

        return self.__get_data('$GPRMC', 'gprmc')

    def set_profile(self):
        """
        Railway profile setter. Uses ProfileReader class.

        Returns
        -------
        pd.DataFrame
            ProfileReader object
        """

        profile = ProfileReader(profile=self.file, rus=self.rus)
        return profile

    def headers_loader(self):
        """
        Loading headers for result table from the .json file.
        """

        try:
            with open('data' + os.sep + 'headers.json', encoding='utf-8') as h:
                headers = json.load(h)
                return headers
        except FileNotFoundError:
            showerror('No data', 'Profile data does not have column names')

    def get_table(self, array2d, data_header=None):
        """
        Result table constructor.

        Parameters
        ----------
        array2d : pd.DataFrame
            Preprocessed table
        data_header : str
            Header's tag

        Returns
        -------
        df : pd.DataFrame
            Formatted and transformed experimental data
        """
        headers = self.headers_loader()

        if data_header == 'gprmc':
            df = pd.DataFrame(array2d, columns=headers['gprmc'])
            utc_time = self.parse_time(df['utc']).apply(self.to_time)
            df['utc'] = (utc_time - utc_time[0]).dt.total_seconds()
            df['date'] = self.parse_date(df['date'])
            df['Speed'] = self.knots_to_mps(df['Speed'])
            df['Latitude'] = self.coord_processing(df['Latitude'], df['Lat_dir'])
            df['Longitude'] = self.coord_processing(df['Longitude'], df['Long_dir'])

            if self.rus:
                trans_dict = {tag[0]: tag[1] for tag in zip(headers['gprmc'], headers['gprmc_rus'])}
                df.rename(trans_dict, axis=1, inplace=True)
                if self.informative:
                    df.drop(['Отклонение_от_курса',
                             'Магнитное_отклонение',
                             'Индикатор_системы позиционирования',
                             '*xx'], axis=1, inplace=True)
            else:
                df.rename({'Speed': 'Speed, mps',
                           'utc': 'time, s'}, axis=1, inplace=True)
                if self.informative:
                    df.drop(['track_true', 'vat_dir', 'mode_ind', '*xx'], axis=1, inplace=True)
            df['Время'] = np.arange(0, len(df) * 0.2, 0.2)

        elif data_header == 'gpgga':
            df = pd.DataFrame(array2d, columns=headers['gpgga'])
            df['alt'] = df['alt'].astype('float32')

        return df

    @staticmethod
    def filt_truncate(dataframe, filter_value):
        """
        If filter is used, return sparse experimental dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Experimental table
        filter_value : int
            Decimation frequency

        Returns
        -------
        dataframe
            Sparse version of the result table
        """

        if filter_value:
            try:
                if len(filter_value) != 2:
                    raise IndexError
                if filter_value[0] and filter_value[1]:
                    a, b = filter_value[0], filter_value[1]
                    dataframe = dataframe[(dataframe['Время'] < a) & dataframe['Время'] > b]
                elif filter_value[0] and not filter_value[1]:
                    a = filter_value[0]
                    dataframe = dataframe[dataframe['Время'] < a]
                elif not filter_value[0] and filter_value[1]:
                    b = filter_value[1]
                    dataframe = dataframe[dataframe['Время'] > b]
                else:
                    pass

            except IndexError:
                print('Пожалуйста, задайте границы фильтра по шаблону!')

            finally:
                return dataframe

    @staticmethod
    def to_time(time_str):
        """
        To-time dataframe convertor.

        Parameters
        ----------
        time_str : pd.Series
            String that originally contains time info

        Returns
        -------
        pd.Series
            Hours:Minutes:Seconds:milliseconds format for time
        """

        return dt.datetime.strptime(time_str, '%H:%M:%S:%f')

    @staticmethod
    def parse_date(column):
        """
        To-datetime dataframe convertor

        Parameters
        ----------
        column : pd.Series
            Column name on which the transformation is needed to be produced

        Returns
        -------
        pd.Series
            Column on which the transformation has been produced
        """

        return column.astype('datetime64').dt.strftime('%d-%m-%Y')

    @classmethod
    def parse_time(cls, column):
        """
        UTS datetime convertor

        Parameters
        ----------
        column : pd.Series
            Column name on which the transformation is needed to be produced

        Returns
        -------
        pd.Series
            Column on which the conversion has been produced
        """

        column = pd.to_datetime(column, unit='s')
        plus_timezone = dt.timedelta(hours=cls.timezone)
        return (column + plus_timezone).dt.strftime('%H:%M:%S:%f')

    @staticmethod
    def knots_to_mps(column):
        """
        Knots to meter per seconds convertor.

        Parameters
        ----------
        column : pd.Series
            Column name on which the transformation is needed to be produced

        Returns
        -------
        pd.Series
            Column on which the conversion has been produced
        """
        return (column.values.astype('float32') * 0.514444).round(2)

    @staticmethod
    def coord_processing(coord, direction):
        """
        The decimal format of degree transformer and coordinate format reducer.

        Parameters
        ----------
        coord : pd.DataFrame, pd.Series, np.array
            A pair of coordinates
        direction : str
            The direction towards which a railway car is moving:
            'N' for North
            'S' for South
            'E' for East
            'W' for West

        Returns
        -------
        float:
            Coordinates in decimal format
        """
        return decimal_degrees(*dms(coord, direction))

    @staticmethod
    def slopes_in_main(arr, auxil):
        """
        Railway slope calculator.

        Parameters
        ----------
        arr : pd.DataFrame
            Initial dataframe
        auxil : pd.DataFrame
            Auxiliary dataframe
        Returns
        -------
        tuple(list, list, float)
            (Railway peg list, railway slope list, length of the railway peg list)
        """
        pic, slp = [], []
        for lat, lon in zip(arr['Широта'] * -1, arr['Долгота'] * -1):
            for lat1, lon1, lat2, lon2, pc, sl in auxil[['Широта',
                                                         'Долгота',
                                                         'Широта_след.',
                                                         'Долгота_след.',
                                                         'Пикет',
                                                         'Уклон']].to_numpy():
                if lat1 >= lat >= lat2:
                    pic.append(pc)
                    slp.append(sl)
                    break
        return pic, slp, len(pic)
