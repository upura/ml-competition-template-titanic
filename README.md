ml-competition-template-titanic
===
[Kaggle Titanic](https://www.kaggle.com/c/titanic) example of my own.

You can get the score = 0.78468 at the version of 2018-12-15.

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
