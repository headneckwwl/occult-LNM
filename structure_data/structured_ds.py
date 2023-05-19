
import os
from typing import List, Union

import numpy as np
import pandas as pd
import torch.utils.data as data

from _algo.utils.about_log import logger


class StructuredDataset(data.Dataset):
    def __init__(self, *data_files, keys: Union[str, List[str]] = None, mode='joint', exclude: List[str] = None,
                 label: str = None, fillna: float = None, label_mapping_func=None,
                 as_onehot: bool = False, n_classes: int = None):
        assert mode in ['joint', 'isolate']
        self.mode = mode
        assert not (as_onehot and n_classes is None)
        self.as_onehot = as_onehot
        self.n_classes = n_classes
        assert all(os.path.exists(data_file) and os.path.isfile(data_file)
                   for data_file in data_files), r'所有的数据必须存在！'
        self.data_files = data_files
        self.data = [pd.read_csv(d) if d.endswith('.csv') else pd.read_excel(d)
                     for d in data_files]

        # 填充获取抛弃nan的数据。
        self.fillna = fillna
        if fillna is None:
            self.data[0] = self.data[0].dropna(axis=0, how='any')
        else:
            self.data[0] = self.data[0].fillna(fillna)
        # 分析可能的key，或者指定拼接的字段
        if keys is None:
            keys = set()
            for d in self.data:
                keys |= set(d.columns)
            self.keys = list(keys)
            logger.info(f'keys参数没有设置，推测可能是：{self.keys}')
        else:
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            self.keys = list(set(keys))
        # 检查key是否唯一
        self.uk = None
        self.check_unique()
        # 所有数据中不参与训练的字段
        if exclude is None:
            self.exclude = self.keys
        else:
            self.exclude = set(exclude) | set(self.keys)

        # 获取标签数据，并且从原始数据中删除。
        self.label_column = [label]
        self.label = None
        if label is not None:
            self.label = self.data[0][self.keys + self.label_column].copy(deep=True)
            if label_mapping_func is not None:
                self.label[label] = self.label[label].map(label_mapping_func)
            self.data[0] = self.data[0].drop(self.label_column, axis=1)

        # 计算每个数据剩余的训练/验证数据，并且获取特征长度。
        self.columns = [[c for c in d.columns if c not in self.exclude]
                        for d in self.data]
        self.feat_len = [len(c) for c in self.columns]
        self._X = None
        self._y = None

    def check_unique(self):
        for idx, d in enumerate(self.data):
            uk = [tuple(l) for l in np.array(d[self.keys])]
            assert len(uk) == len(set(uk)), f"{self.data_files[idx]} 的key并不唯一！"
            if idx == 0:
                self.uk = uk

    @staticmethod
    def get_data(df, keys, values, columns, fillna, dtype=np.float32):
        condition = None
        for key, value in zip(keys, values):
            if condition is None:
                condition = df[key] == value
            else:
                condition &= df[key] == value
        if not df[condition][columns].empty:
            return np.reshape(np.array(df[condition][columns].astype(dtype)), (-1,))
        elif fillna is not None:
            return np.array([fillna] * len(columns), dtype=dtype)
        else:
            return None

    def construct_data(self):
        X = [[StructuredDataset.get_data(df, self.keys, values, columns, self.fillna)
              for df, columns in zip(self.data, self.columns)]
             for values in self.uk]
        self.uk = [v for d, v in zip(X, self.uk) if all([x is not None for x in d])]
        if self.mode == 'joint':
            X = [[np.concatenate(d)] for d in X if all([x is not None for x in d])]
        else:
            X = [d for d in X if all([x is not None for x in d])]
        self._X = X
        if self.label is not None:
            self._y = [StructuredDataset.get_data(self.label, self.keys, values, self.label_column, self.fillna,
                                                  dtype=np.int64 if self.n_classes is not None else np.float32)
                       for values in self.uk]

    @property
    def X(self):
        if self._X is None:
            self.construct_data()
        return self._X

    @property
    def y(self):
        return self._y

    def __getitem__(self, item):
        if self.y is not None:
            if self.as_onehot:
                return self.X[item], np.eye(self.n_classes, dtype=np.float32)[np.squeeze(self.y[item])]
            else:
                return self.X[item], self.y[item]
        return self.X[item], None

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    ds = StructuredDataset('bc_data.csv', fillna=0, label='diagnosis', keys='id',
                           exclude=['diagnosis'], mode='isolate',
                           label_mapping_func=lambda x: 1 if x == 'M' else 0)
    print(ds[0][1].shape)
