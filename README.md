ml-competition-template-titanic
===
- [Kaggle Titanic](https://www.kaggle.com/c/titanic) example of my own, inspired by [flowlight0's repo](https://github.com/flowlight0/talkingdata-adtracking-fraud-detection).
- You can get the score = 0.76555 at the version of 2018-12-28.
- Japanese article can be seen [here](https://upura.hatenablog.com/entry/2018/12/28/225234).

# Structures
```
.
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── output
├── features
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   └── logger.py
├── models
│   └── lgbm.py
├── notebooks
│   └── eda.ipynb
├── scripts
│   └── convert_to_feather.py
├── utils
│   └── __init__.py
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── run.py
└── tox.ini
```
# Commands

## Change data to feather format

```
python scripts/convert_to_feather.py
```

## Create features

```
python features/create.py
```

## Run LightGBM

```
python run.py
```

## flake8

```
flake8 .
```
