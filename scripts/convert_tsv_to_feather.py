import pandas as pd

target = [
    'train',
    'test',
    'weather',
    'nied_oyama',
    'nightley',
    'hotlink'
]

for t in target:
    (pd.read_table('./data/input/' + t + '.tsv', encoding="utf-8"))\
        .to_feather('./data/input/' + t + '.feather')
