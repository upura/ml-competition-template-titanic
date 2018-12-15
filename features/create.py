import pandas as pd
import math
import datetime

from features.base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class Datetime(Feature):
    def create_features(self):
        self.train['datetime'] = train['datetime']
        self.test['datetime'] = test['datetime']


class Year(Feature):
    def create_features(self):
        self.train['year'] = pd.to_datetime(train['datetime']).dt.year
        self.test['year'] = pd.to_datetime(test['datetime']).dt.year


class Month(Feature):
    def create_features(self):
        self.train['month'] = pd.to_datetime(train['datetime']).dt.month
        self.test['month'] = pd.to_datetime(test['datetime']).dt.month


class Day(Feature):
    def create_features(self):
        self.train['day'] = pd.to_datetime(train['datetime']).dt.day
        self.test['day'] = pd.to_datetime(test['datetime']).dt.day


class Dow(Feature):
    def create_features(self):
        self.train['dow'] = pd.to_datetime(train['datetime']).dt.weekday
        self.test['dow'] = pd.to_datetime(test['datetime']).dt.weekday


class Date_sin(Feature):
    def create_features(self):
        self.train['date_sin'] = [
            round(math.sin(math.radians(
                ((datetime.datetime(2016, 12, 31)
                    - datetime.datetime.strptime(
                        date_str, '%Y-%m-%d'
                    )).days/365)*360
            )), 5) for date_str in train['datetime']
        ]
        self.test['date_sin'] = [
            round(math.sin(math.radians(
                ((datetime.datetime(2016, 12, 31)
                    - datetime.datetime.strptime(
                        date_str, '%Y-%m-%d'
                    )).days/365)*360
            )), 5) for date_str in test['datetime']
        ]


class Date_cos(Feature):
    def create_features(self):
        self.train['date_cos'] = [
            round(math.cos(math.radians((
                    (datetime.datetime(2016, 12, 31)
                        - datetime.datetime.strptime(
                            date_str, '%Y-%m-%d'
                        )).days/365
                )*360)), 5) for date_str in train['datetime']
            ]
        self.test['date_cos'] = [
            round(math.cos(math.radians((
                    (datetime.datetime(2016, 12, 31)
                        - datetime.datetime.strptime(
                            date_str, '%Y-%m-%d'
                        )).days/365
                )*360)), 5) for date_str in test['datetime']
            ]


class Month_sin(Feature):
    def create_features(self):
        self.train['month_sin'] = [
            round(
                math.sin(math.radians(m*360/12)), 5
            ) for m in pd.to_datetime(train['datetime']).dt.month
        ]
        self.test['month_sin'] = [
            round(
                math.sin(math.radians(m*360/12)), 5
            ) for m in pd.to_datetime(test['datetime']).dt.month
        ]


class Month_cos(Feature):
    def create_features(self):
        self.train['month_cos'] = [
            round(
                math.cos(math.radians(m*360/12)), 5
            ) for m in pd.to_datetime(train['datetime']).dt.month
        ]
        self.test['month_cos'] = [
            round(
                math.cos(math.radians(m*360/12)), 5
            ) for m in pd.to_datetime(test['datetime']).dt.month
        ]


class Day_sin(Feature):
    def create_features(self):
        self.train['day_sin'] = [
            round(
                math.sin(math.radians(d*360/31)), 5
            ) for d in pd.to_datetime(train['datetime']).dt.day
        ]
        self.test['day_sin'] = [
            round(
                math.sin(math.radians(d*360/31)), 5
            ) for d in pd.to_datetime(test['datetime']).dt.day
        ]


class Day_cos(Feature):
    def create_features(self):
        self.train['day_cos'] = [
            round(
                math.cos(math.radians(d*360/31)), 5
            ) for d in pd.to_datetime(train['datetime']).dt.day
        ]
        self.test['day_cos'] = [
            round(
                math.cos(math.radians(d*360/31)), 5
            ) for d in pd.to_datetime(test['datetime']).dt.day
        ]


class Dow_sin(Feature):
    def create_features(self):
        self.train['dow_sin'] = [
            round(
                math.sin(math.radians((w)*360/7)), 5
            ) for w in pd.to_datetime(train['datetime']).dt.weekday
        ]
        self.test['dow_sin'] = [
            round(
                math.sin(math.radians((w)*360/7)), 5
            ) for w in pd.to_datetime(test['datetime']).dt.weekday
        ]


class Dow_cos(Feature):
    def create_features(self):
        self.train['dow_cos'] = [
            round(
                math.cos(math.radians((w)*360/7)), 5
            ) for w in pd.to_datetime(train['datetime']).dt.weekday
        ]
        self.test['dow_cos'] = [
            round(
                math.cos(math.radians((w)*360/7)), 5
            ) for w in pd.to_datetime(test['datetime']).dt.weekday
        ]


class Holiday(Feature):
    def create_features(self):
        self.train['holiday'] = [
            1 if (x == 5 or x == 6) else 0
            for x in pd.to_datetime(train['datetime']).dt.weekday
        ]
        self.test['holiday'] = [
            1 if (x == 5 or x == 6) else 0
            for x in pd.to_datetime(test['datetime']).dt.weekday
        ]


class Park(Feature):
    def create_features(self):
        self.train['park'] = train['park']
        self.test['park'] = test['park']


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
