import numpy as np
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

conn = pymysql.connect(
    host="192.168.7.93",
    user="quantchina",
    password="zMxq7VNYJljTFIQ8",
    database="wind",
    charset="gbk"
)
# 读取个股的换手率以及市值
cursor = conn.cursor()
query = """
    SELECT TRADE_DT, S_INFO_WINDCODE, S_DQ_TURN, S_VAL_MV
    FROM ASHAREEODDERIVATIVEINDICATOR
    WHERE TRADE_DT in (select TRADE_DT from AINDEXEODPRICES where S_INFO_WINDCODE='000985.CSI' and 
        TRADE_DT between '20230101' and '20240201') and 
        S_INFO_WINDCODE in (
            SELECT S_CON_WINDCODE
            FROM AINDEXCSIALLINDWEIGHT
            where TRADE_DT='20230104'
            ORDER BY S_CON_WINDCODE
    )
    ORDER BY TRADE_DT, S_INFO_WINDCODE
"""
cursor.execute(query)

data = cursor.fetchall()
df = pd.DataFrame(data, columns=['date', 'code', 'turnover', 'vals'])
df['turnover'] = df['turnover'].astype(float) / 100

# 读取个股的行业信息
query = """
    SELECT WIND_CODE, WIND_IND_CODE
    FROM ASHAREINDUSTRIESCLASS
    WHERE CUR_SIGN=1 and WIND_CODE in (
        SELECT S_CON_WINDCODE
        FROM AINDEXCSIALLINDWEIGHT
        where TRADE_DT='20230104'
    )
    ORDER BY WIND_CODE
"""
cursor.execute(query)

ind_data = cursor.fetchall()
ind = pd.DataFrame(ind_data, columns=['code', 'industry'])
ind.set_index('code', inplace=True)
# 行业使用Wind定义的11个一级行业
for i in range(len(ind)):
    ind.iloc[i, 0] = (ind.iloc[i, 0])[:4]

# 读取个股的收益率数据
query = """
    SELECT TRADE_DT, S_INFO_WINDCODE, S_DQ_PCTCHANGE
    FROM ASHAREEODPRICES    
    WHERE S_INFO_WINDCODE in (
        SELECT S_CON_WINDCODE
        FROM AINDEXCSIALLINDWEIGHT
        WHERE TRADE_DT='20230104'
    ) and TRADE_DT between '20230101' and '20240201'
    ORDER BY TRADE_DT, S_INFO_WINDCODE 
"""
cursor.execute(query)
data = cursor.fetchall()

returns = pd.DataFrame(data, columns=['date', 'code', 'pct_change'])
returns['pct_change'] = returns['pct_change'].astype(float) / 100

cursor.close()
conn.close()

# 合并取得的所有数据
merge = pd.concat([df, returns['pct_change']], axis=1)
merge.dropna(inplace=True)


# 返回列表，其中每个元素是每天中性化的换手率因子收益率
def stability(merge):
    factor_r = []
    for date in tqdm(merge.date.unique(), desc="Processing"):
        scaler = StandardScaler()

        df_ols = (merge[merge.date == date]).iloc[:, 1:]
        df_ols.set_index('code', inplace=True)
        df_ols = pd.merge(df_ols, ind, left_index=True, right_index=True, how='inner')
        df_ols.vals = scaler.fit_transform(df_ols.vals.to_numpy().reshape(-1, 1))

        # 对11个行业设置10个虚拟变量
        dummies = pd.get_dummies(df_ols.industry)
        dummies.drop(['6260'], axis=1, inplace=True)
        df_dummies = pd.concat([df_ols, dummies], axis=1)
        y = df_dummies.turnover.to_numpy()
        X = sm.add_constant(df_dummies.drop(['turnover', 'industry', 'pct_change'], axis=1).to_numpy())
        results = sm.OLS(y, X).fit()
        # 换手率对市值和行业中性化
        resid = results.resid

        # 收益率对中性化后的换手率因子回归
        r = df_dummies['pct_change'].to_numpy()
        res = sm.OLS(r, resid).fit()
        factor_r.append(res.params[0])

    return factor_r
