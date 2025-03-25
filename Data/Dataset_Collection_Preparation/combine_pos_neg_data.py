import pandas as pd
pos_data = pd.read_csv('ires_pos.csv', index_col = 'Index', encoding='gbk')
neg_data = pd.read_csv('ires_neg.csv', index_col = 'Index', encoding='gbk')
pos_data['label'] = 1
neg_data['label'] = 0
data = pd.concat([pos_data, neg_data], axis = 0).drop_duplicates('Sequence', keep = False)
data['Sequence_174'] = [s[18:-20] for s in data.Sequence]
print(pos_data.shape, neg_data.shape, pos_data.shape[0] + neg_data.shape[0], data.shape)
data.to_csv('data_pos_neg.csv', index = True)
