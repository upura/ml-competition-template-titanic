import pandas as pd
import numpy as np
import re as re

from features.base import Feature, get_arguments, generate_features

Feature.dir = 'features'


class Pclass(Feature):
    def create_features(self):
        self.train['Pclass'] = train['Pclass']
        self.test['Pclass'] = test['Pclass']


class Sex(Feature):
    def create_features(self):
        self.train['Sex'] = train['Sex'].replace(['male', 'female'], [0, 1])
        self.test['Sex'] = test['Sex'].replace(['male', 'female'], [0, 1])


class FamilySize(Feature):
    def create_features(self):
        self.train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
        self.test['FamilySize'] = test['Parch'] + test['SibSp'] + 1


class Embarked(Feature):
    def create_features(self):
        self.train['Embarked'] = train['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)
        self.test['Embarked'] = test['Embarked'] \
            .fillna(('S')) \
            .map({'S': 0, 'C': 1, 'Q': 2}) \
            .astype(int)


class Fare(Feature):
    def create_features(self):
        data = train.append(test)
        fare_mean = data['Fare'].mean()
        self.train['Fare'] = pd.qcut(
            train['Fare'].fillna(fare_mean),
            4,
            labels=False
        )
        self.test['Fare'] = pd.qcut(
            test['Fare'].fillna(fare_mean),
            4,
            labels=False
        )


class Age(Feature):
    def create_features(self):
        data = train.append(test)
        age_mean = data['Age'].mean()
        age_std = data['Age'].std()
        self.train['Age'] = pd.qcut(
            train['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )
        self.test['Age'] = pd.qcut(
            test['Age'].fillna(
                np.random.randint(age_mean - age_std, age_mean + age_std)
            ),
            5,
            labels=False
        )


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


class Title(Feature):
    def create_features(self):
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        train['Title'] = train['Name'] \
            .apply(get_title) \
            .replace([
                'Lady',
                'Countess',
                'Capt',
                'Col',
                'Don',
                'Dr',
                'Major',
                'Rev',
                'Sir',
                'Jonkheer',
                'Dona'
            ], 'Rare') \
            .replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
        train['Title'] = train['Name'].map(title_mapping).fillna(0)
        test['Title'] = test['Name'] \
            .apply(get_title) \
            .replace([
                'Lady',
                'Countess',
                'Capt',
                'Col',
                'Don',
                'Dr',
                'Major',
                'Rev',
                'Sir',
                'Jonkheer',
                'Dona'
            ], 'Rare') \
            .replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
        test['Title'] = test['Title'].map(title_mapping).fillna(0)

        self.train['Title'] = train['Title']
        self.test['Title'] = test['Title']


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
