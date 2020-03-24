import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame


def makegraph(row_begin,row_end):
    data_train = pd.read_csv("data/HousePrices/test.csv")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    # data_train.columns = []
    pd.set_option('display.expand_frame_repr', False)
    print(data_train[row_begin:row_end])

    # print(data_train)
    # print([column for column in data_train])
    # data_train
