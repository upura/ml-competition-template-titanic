import pandas as pd

target = [
    'jorudan'
]

for t in target:
    (pd.read_table('./data/input/' + t + '.zip', encoding="utf-8"))\
        .to_feather('./data/input/' + t + '.feather')
