import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
seed = 1337

data = pd.read_csv('data_pos_neg.csv', index_col = 'Index').reset_index(drop = True)
x = range(len(data))
train_idx, test_idx = train_test_split(x, random_state = seed, shuffle = True, stratify = data['label'], test_size = 0.1)
print(test_idx[:10])

train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]
print(train_data.shape, test_data.shape)
print(Counter(train_data.label), Counter(test_data.label))
test_data.head()

train_data.sample(frac = 1).to_csv('train_data_pos_neg_split0.1.csv', index = False)
test_data.sample(frac = 1).to_csv('test_data_pos_neg_split0.1.csv', index = False)
