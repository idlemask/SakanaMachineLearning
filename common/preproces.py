import pandas as pd  # 数据分析
import numpy as np  # 科学计算


# 去除含有空数据的样本
def deleteNaNData(df):
    # 参数检查
    if not isinstance(df, pd.DataFrame):
        raise TypeError('bad operand type,parameter must be a DataFrame')
    return df[df[df.columns].isin(['NaN'])]


