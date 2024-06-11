import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def ewma(window, half_life):
    """
    给出指数移动平均的权重
    :param window:    窗口长度
    :param half_life: 半衰期
    """
    lamb = 0.5 ** (1 / half_life)
    weights = np.array([lamb ** (window - i) for i in range(window)])
    return weights


def get_adjust_dt(date_ls, freq):
    """
    根据给定的日期区间和频率获取调仓日期，即每月或每周的第一个交易日
    :param date_ls: 一段时间内的所有交易日
    :param freq:    调仓频率
    """
    result = []
    if freq == 'day':
        result = date_ls

    elif freq == 'week':
        for i in range(len(date_ls)):
            if date_ls[i].weekday() == 0:
                result.append(date_ls[i])

    elif freq == 'month':
        for i in range(len(date_ls)):
            if i == 0:
                result.append(date_ls[i])
            elif date_ls[i - 1].month < date_ls[i].month:
                result.append(date_ls[i])

    result = [datetime.strftime(date, '%Y%m%d') for date in result]
    return result


def max_drawdown(cml_rets):
    """
    计算投资组合的最大回撤，最大回撤时长
    :param cml_rets: 组合的累计收益
    """
    n = len(cml_rets)
    res = 0
    before = after = 0
    a = cml_rets[0]
    for i in range(1, n):
        b = cml_rets[i]
        if (1 - b / a) > res:
            res = 1 - b / a
            after = i
        if b > a:
            a = b
            before = i
    return res, after - before


def neutralize(df, industry):
    """
    将因子对市值，行业中性化
    :params df:         各列依次为日期，股票代码，需要中性化的因子，市值
    :params industry:   含个股行业的数据框
    :return neutralize: 返回列表，中性化后的因子值
    """
    result = np.array([])
    # OLS取残差实现中性化处理
    for date in tqdm(df.date.unique(), desc="Processing"):
        scaler = StandardScaler()
        # 把日期列剔除，设置股票代码为行索引
        df_ols = (df[df.date == date]).iloc[:, 1:]
        ind_ols = (industry[industry.date == date]).iloc[:, 1:]
        df_ols.set_index('code', inplace=True)
        ind_ols.set_index('code', inplace=True)
        # 合并数据框，Z-Score标准化
        df_ols = pd.merge(df_ols, ind_ols, left_index=True, right_index=True, how='inner')
        df_ols['fac_vals'] = scaler.fit_transform(df_ols.fac_vals.to_numpy().reshape(-1, 1))
        df_ols['vals'] = scaler.fit_transform(df_ols.vals.to_numpy().reshape(-1, 1))
        # 对11个一级行业添加10个虚拟变量
        dummies = pd.get_dummies(df_ols.industry)
        dummies.drop(['6260'], axis=1, inplace=True)
        df_dummies = pd.concat([df_ols, dummies], axis=1)
        # 因子中性化
        y = df_dummies.fac_vals.to_numpy()
        X = sm.add_constant(df_dummies.drop(['fac_vals', 'industry'], axis=1).to_numpy())
        model = sm.OLS(y, X).fit()
        result = np.hstack((result, model.resid))

    return result


def group_divide(factor, daily_return, group=5):
    """分层后每层等权计算组合收益"""
    cml = (1 + daily_return).cumprod().T
    cml['fac'] = factor.mean().T
    cml['group'] = pd.qcut(cml['fac'], q=group, labels=range(1, group + 1))
    tmp = cml.groupby('group').mean().iloc[:, :-1]

    return tmp


def factor_group(factor, fac_all, returns, hist_ret, freq, group=5):
    """
    分层法测试单因子单调性
    :param factor:   当日指数成分股的因子值，有三列date, code, factor values
    :param fac_all:  所有在指数中出现过的股票的因子值，索引为date，列名为code，值为factor values
    :param returns:  当日指数个股的收益率，有三列date, code, daily return
    :param hist_ret: 所有在指数中出现过的股票的日收益率，索引为date，列名为code，值为daily return
    :param freq:     调仓频率
    :param group:    分组个数
    """
    trade_dt = returns.date.unique()
    labels = [i for i in range(1, group + 1)] + ['long_short']
    # 按分组和日期储存结果
    result = pd.DataFrame(1, index=labels, columns=[trade_dt[0]])

    # 每次获得一期的累计收益率，与上一期末的累积收益相乘，得到总的累计收益
    if freq == 'day':
        for i in range(len(trade_dt) - 1):
            codes = factor[factor.date == trade_dt[i]].code
            fac_df = pd.pivot_table(factor[factor.date == trade_dt[i]], values='fac_vals', index='date', columns='code')
            rets = hist_ret.loc[[trade_dt[i + 1]], codes]
            cml = rets.T
            cml['fac'] = fac_df.mean().T
            # 通过因子值分层计算组合收益
            cml['group'] = pd.qcut(cml['fac'], q=group, labels=range(1, group + 1))
            tmp = cml.groupby('group').mean().iloc[:, :-1]
            tmp = pd.DataFrame(tmp, index=range(1, group + 1))
            tmp.loc['long_short'] = tmp.loc[1] - tmp.loc[group]
            tmp = (1 + tmp).cumprod(axis=1)
            result = pd.concat([result, tmp.apply(lambda x: x * result.iloc[:, -1])], axis=1)

    else:
        dates = pd.to_datetime(trade_dt)
        # 获取每周或每月的第一个交易日
        adj_dt = get_adjust_dt(dates, freq)
        # 加入结束日期，得到日期区间的完整分割
        if adj_dt[-1] != trade_dt[-1]:
            adj_dt.append(trade_dt[-1])
        n = len(adj_dt)
        for i in range(n - 1):
            start, end = adj_dt[i], adj_dt[i + 1]
            codes = factor[factor.date == start].code
            fac_df = (fac_all.loc[start:end, codes])[:-1]  # 一期内的因子值
            rets = (hist_ret.loc[start:end, codes])[1:]  # 一期内的日收益率
            cml = (1 + rets).cumprod().T - 1
            cml['fac'] = fac_df.mean().T
            cml['group'] = pd.qcut(cml['fac'], q=group, labels=range(1, group + 1))
            tmp = cml.groupby('group').mean().iloc[:, :-1]
            tmp = pd.DataFrame(tmp, index=range(1, group + 1))
            tmp.loc['long_short'] = -tmp.loc[1] + tmp.loc[group]
            result = pd.concat([result, (1 + tmp).apply(lambda x: x * result.iloc[:, -1])], axis=1)

    return result


def plot_by_group(df_group):
    """画出不同层次的累计收益随时间变化图"""
    group = df_group.shape[0] - 1
    for i in range(0, group):
        plt.plot(df_group.iloc[i, :], linewidth=1.5, alpha=0.8, label=str(i + 1))
    plt.plot(df_group.iloc[-1, :], linewidth=1.5, alpha=0.8, label='long short')

    plt.title('Daily Portfolio Value Over Time ')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 读取数据
    names = pd.read_parquet('names.parquet')
    all_df = pd.read_parquet('factors(2017-2023).parquet')
    ind = pd.read_parquet('industry(2017-2023).parquet')
    hist_ret = pd.read_parquet('hist returns(17-23).parquet')

    # 通过pandas合成数据
    factors = pd.merge(names, all_df, on=['date', 'code'], how='left')
    factors = factors.rename(columns={'turnover': 'fac_vals'})
    returns = pd.merge(names, hist_ret, on=['date', 'code'], how='left')
    hist_ret = pd.pivot_table(hist_ret, index='date', columns='code', values='pct_change')
    all_factors = pd.pivot_table(all_df.drop(columns='vals'), index='date', columns='code', values='turnover')

    factor_df = factors.drop(columns='vals')
    factor_df['fac_vals'] = neutralize(factors, ind)

    grouped = factor_group(factor_df, all_factors, returns, hist_ret, 'day', 5)
    plot_by_group(grouped)
