import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def ewma(window, half_life):
    """给出指数移动平均的权重"""
    lamb = 0.5 ** (1 / half_life)
    weights = np.array([lamb ** (window - i) for i in range(window)])
    return weights


def calc_beta(returns, hist_ret, index_rets):
    """
    计算Beta
    :param returns:  指数成分股的日收益率，有三列date, code, daily returns
    :param hist_ret: 所有在指数中出现过的股票的日收益率，索引为date，列名为code，值为收益率
    :param index_rets:  市场组合的日收益率
    """
    trade_dt = hist_ret.index
    dt = returns.date.unique()
    window, half_life = 252, 63
    W = ewma(window, half_life)
    beta_list = []

    for i in tqdm(range(len(dt))):
        end_ind = np.where(trade_dt == dt[i])[0] - 1
        end = trade_dt[end_ind[0]]
        start = trade_dt[end_ind[0] - window + 1]

        mp = index_rets.loc[start:end, :]
        X = sm.add_constant(mp)
        X_mat = X.to_numpy()
        codes = returns[returns.date == dt[i]].code
        tmp = hist_ret.loc[start:end, codes]
        # 有缺失值和没有缺失值的数据分开处理
        y1, W1 = tmp.dropna(axis=1), np.diag(W)
        params = np.linalg.pinv(X_mat.T @ W1 @ X_mat) @ X_mat.T @ W1 @ y1
        beta1 = params.iloc[1, :]

        lack = tmp.columns[tmp.isna().any()]
        beta2_list = []
        for code in lack:
            tmp2 = tmp.loc[:, code]
            tmp2 = pd.concat([X, tmp2], axis=1)
            tmp2['weights'] = W
            tmp2.dropna(inplace=True)
            # 有NaN则删除后用剩余值计算，少于63天不计算
            if len(tmp2) < half_life:
                continue
            y2 = tmp2[code]
            W2 = np.diag(tmp2['weights'])
            X2 = tmp2.iloc[:, :2].to_numpy()
            params = np.linalg.pinv(X2.T @ W2 @ X2) @ X2.T @ W2 @ y2
            beta2_list.append(params[1])

        beta2 = pd.Series(beta2_list, index=lack)
        beta_list.append(pd.concat([beta1, beta2]).sort_index())
    # 返回一个储存Beta值的数据框
    return pd.concat(beta_list, axis=0)


def stability(df, industry, rets):
    """
    得到每日的因子收益时间序列
    :params df:        含所有因子值的数据框
    :params industry:  含个股行业的数据框
    :params rets:      含所有个股收益率的数据框
    :return stability: 返回列表，其中每个元素是每天中性化的换手率因子收益率
    """
    result = []
    t_values = []
    dates = df.date.unique()
    # 每日对因子中性化处理，OLS得到因子收益
    for i in tqdm(range(len(dates) - 1), desc="Processing"):
        scaler = StandardScaler()

        df_ols = (df[df.date == dates[i]]).iloc[:, 1:]
        codes = df_ols.code
        ind_ols = (industry[industry.date == dates[i]]).iloc[:, 1:]
        df_ols.set_index('code', inplace=True)
        ind_ols.set_index('code', inplace=True)
        df_ols = pd.merge(df_ols, ind_ols, left_index=True, right_index=True, how='inner')
        df_ols.vals = scaler.fit_transform(df_ols.vals.to_numpy().reshape(-1, 1))
        # df_ols.beta = scaler.fit_transform(df_ols.beta.to_numpy().reshape(-1, 1))
        # 对11个一级行业添加10个虚拟变量
        dummies = pd.get_dummies(df_ols.industry)
        dummies.drop(['6260'], axis=1, inplace=True)
        df_dummies = pd.concat([df_ols, dummies], axis=1)
        # 因子中性化
        y = df_dummies.turnover.to_numpy()
        X = sm.add_constant(df_dummies.drop(['turnover', 'industry'], axis=1).to_numpy())
        results = sm.OLS(y, X).fit()
        resid = results.resid

        # 收益率对中性化后的因子回归
        ep_rets = rets.loc[dates[i + 1], codes]
        df_res = pd.DataFrame({'y': ep_rets.values, 'x': resid}, index=ep_rets.index)
        df_res['b'] = 1
        df_res.dropna(inplace=True)
        res = sm.OLS(df_res['y'], df_res[['b', 'x']]).fit()
        result.append(res.params[1])
        t_values.append(res.tvalues[1])

    return result, t_values


if __name__ == '__main__':
    # 读取数据
    factors = pd.read_parquet('factors.parquet')
    ind = pd.read_parquet('industry(2017-2023).parquet')
    hist_ret = pd.read_parquet('hist returns(17-23).parquet')
    port_mv = pd.read_parquet('CSI all index(17-23).parquet')

    # 通过pandas合成数据


    # beta = calc_beta(returns, hist_ret, port_mv)
    # factors['beta'] = beta.values
    factor_r, factor_t = stability(factors, ind, hist_ret)
    print(sum(factor_r)/len(factor_r))
    # 画图查看因子收益稳定性
    plt.plot(factors.date.unique()[:-1], factor_r)
    plt.axhline(y=sum(factor_r)/len(factor_r), color='r', linestyle='--')
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.plot(factors.date.unique()[:-1], factor_t)
    plt.axhline(y=2, color='r', linestyle='--')
    plt.axhline(y=-2, color='r', linestyle='--')
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xticks(rotation=45) 
    plt.tight_layout()
    plt.show()
