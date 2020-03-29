from sklearn import linear_model
import common.preproces as hpp
import pandas as pd  # 数据分析
import numpy as np
import matplotlib.pyplot as plt


def model1():
    # 读取训练集
    dt = pd.read_csv("data/HousePrices/test.csv")
    dss = pd.read_csv("data/HousePrices/sample_submission.csv")
    # 选择特定的列
    dt = dt[['Id', 'LotArea']]
    dtss = pd.merge(dt, dss, on='Id')
    # 预处理
    hpp.deleteNaNData(dtss)
    # 将数据转换成行数所需的格式
    area = np.asarray(dtss['LotArea'].values)
    price = np.asarray(dtss['SalePrice'].values)
    X = []
    Y = []
    # 结果集是向量组
    for i in price:
        Y.append(i)
    # 变量集是个矩阵
    for i in area:
        X.append([i])
    feature = np.array(X)
    target = np.array(Y)
    print(X)
    # 创建线性回归模型对象
    model = linear_model.LinearRegression()
    # 拟合模型
    model.fit(feature, target)
    # 预测结果
    x = np.linspace(0, 50000, num=10000)
    y = model.predict(x.reshape(-1, 1))
    # 画图
    # 预测的点
    plt.scatter(x, y)
    # 设定字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 样本的点
    plt.scatter(area, price)
    # 设定各种名称
    plt.xlabel = ('地块面积')
    plt.set_ylabel('售价')
    plt.title('地块面积和售价的关系')
    # 显示图像
    plt.show()


def model2():
    # 读取训练集
    dt = pd.read_csv("data/HousePrices/test.csv")
    dss = pd.read_csv("data/HousePrices/sample_submission.csv")
    # 选择特定的列
    dt = dt[['Id', 'MSZoning']]
    dt.replace('A', 1 , inplace = True)
    dt.replace('I', 2, inplace = True)
    dt.replace('FV', 3, inplace = True)
    dt.replace('RL', 4, inplace = True)
    dt.replace('RP', 5, inplace = True)
    dt.replace('RM', 6, inplace = True)
    dt.replace('RH', 7, inplace = True)
    dt.replace('C (all)', 8, inplace = True)

    dtss = pd.merge(dt, dss, on='Id')
    # 预处理
    print(dtss)
    hpp.deleteNaNData(dtss)



    # 将数据转换成行数所需的格式
    MSZoning = np.asarray(dtss['MSZoning'].values)
    price = np.asarray(dtss['SalePrice'].values)
    X = []
    Y = []
    # 结果集是向量组
    for i in price:
        Y.append(i)
    # 变量集是个矩阵
    for i in MSZoning:
        X.append([i])
    feature = np.array(X)
    target = np.array(Y)
    print(X)
    # 创建线性回归模型对象
    # model = linear_model.LinearRegression()
    # 拟合模型
    # model.fit(feature, target)
    # 预测结果
    # x = np.asarray(range(1,8))
    # y = model.predict(x.reshape(-1, 1))
    # 画图
    # 预测的点
    # plt.scatter(x, y)
    # 设定字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 样本的点
    plt.scatter(MSZoning, price)
    # 设定各种名称
    # plt.xlabel = ('地块面积')
    # plt.set_ylabel('售价')
    # plt.title('地块面积和售价的关系')
    # 显示图像
    plt.show()