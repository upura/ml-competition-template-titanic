ml-competition-template-titanic
===
[Kaggle Titanic](https://www.kaggle.com/c/titanic) example of my own.

You can get the score = 0.76555 at the version of 2018-12-28.

## Change data to feather format

```
python scripts/convert_to_feather.py
```

## Create features

```
python features/create.py
```

## lightGBM

```
python run.py
```

## flake8

```
flake8 .
```
