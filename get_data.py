import pymysql
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

conn = pymysql.connect(
    host="192.168.7.93",
    user="quantchina",
    password="zMxq7VNYJljTFIQ8",
    database="wind",
    charset="gbk"
)
cursor = conn.cursor()


def get_daily_codes(code, start, end):
    """读取指数每日的成分股代码"""
    query = """
            select a.TRADE_DT, a.S_INFO_WINDCODE
            from ASHAREEODPRICES as a
            where a.TRADE_DT between '{start}' and '{end}' and a.S_INFO_WINDCODE in (
                select S_CON_WINDCODE
                from AINDEXMEMBERS
                where S_INFO_WINDCODE = '{code}' and S_CON_INDATE <= a.TRADE_DT and 
                (S_CON_OUTDATE >= a.TRADE_DT or S_CON_OUTDATE is null)
            )
            order by a.TRADE_DT, a.S_INFO_WINDCODE
            """.format(code=code, start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['date', 'code'])

    df.to_parquet('names(2010-2023).parquet', index=False)


def get_df(code, start, end, filename):
    """读取成分股的换手率以及市值"""
    query = """
        select TRADE_DT, S_INFO_WINDCODE, S_DQ_TURN, S_VAL_MV
        from ASHAREEODDERIVATIVEINDICATOR as a
        where TRADE_DT between '{start}' and '{end}' and 
              S_INFO_WINDCODE in (select distinct S_CON_WINDCODE
                                  from AINDEXMEMBERS
                                  where S_INFO_WINDCODE = '{code}' and S_CON_INDATE <= '{end}' and 
                                        (S_CON_OUTDATE >= '{start}' or S_CON_OUTDATE is null) )
        order by TRADE_DT, S_INFO_WINDCODE
    """.format(code=code, start=start, end=end)
    cursor.execute(query)

    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['date', 'code', 'turnover', 'vals'])
    df['turnover'] = df['turnover'].astype(float) / 100
    df['vals'] = np.log(df['vals'].astype(float))  # 市值取自然对数
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    df.to_parquet(filename, index=False)


def get_industry(code, start, end, filename):
    """读取个股的行业信息"""
    query = """
    select a.TRADE_DT, b.S_INFO_WINDCODE, b.WIND_IND_CODE
    from (select distinct TRADE_DT
          from ASHAREEODPRICES
          where TRADE_DT between '{start}' and '{end}'
          order by TRADE_DT) as a
    left join (select S_INFO_WINDCODE, WIND_IND_CODE, ENTRY_DT, REMOVE_DT
          from ASHAREINDUSTRIESCLASS
          where ENTRY_DT <= '{end}' and (REMOVE_DT >= '{start}' or REMOVE_DT is null) and
          S_INFO_WINDCODE in (select distinct S_CON_WINDCODE
                              from AINDEXMEMBERS
                              where S_INFO_WINDCODE = '{code}' and S_CON_INDATE <= '{end}' and 
                                        (S_CON_OUTDATE >= '{start}' or S_CON_OUTDATE is null))) as b
    on a.TRADE_DT >= b.ENTRY_DT and (b.REMOVE_DT >= a.TRADE_DT or b.REMOVE_DT is null)
    order by a.TRADE_DT, b.S_INFO_WINDCODE
    """.format(code=code, start=start, end=end)
    cursor.execute(query)

    ind_data = cursor.fetchall()
    ind = pd.DataFrame(ind_data, columns=['date', 'code', 'industry'])
    # 使用Wind一级行业代码，取完整代码的前4位
    ind['industry'] = ind['industry'].str[:4]
    ind.to_parquet(filename, index=False)


def get_returns(code, start, end):
    date = datetime.strptime(start, "%Y%m%d")
    # 将日期往前推13个月，以满足252个交易日
    hist = date - timedelta(days=13 * 30)
    hist = hist.strftime("%Y%m%d")

    # 按代码读取股票历史收益数据
    query = """
        select TRADE_DT, S_INFO_WINDCODE, S_DQ_PCTCHANGE
        from ASHAREEODPRICES
        where TRADE_DT between '{hist}' and '{end}' and S_INFO_WINDCODE in (
            select distinct S_CON_WINDCODE
            from AINDEXMEMBERS
            where S_INFO_WINDCODE = '{code}' and S_CON_INDATE <= '{end}' and 
                (S_CON_OUTDATE >= '{start}' or S_CON_OUTDATE is null)
            )
        order by TRADE_DT, S_INFO_WINDCODE
    """.format(code=code, hist=hist, start=start, end=end)

    cursor.execute(query)
    data = cursor.fetchall()
    hist_ret = pd.DataFrame(data, columns=['date', 'code', 'pct_change'])
    hist_ret['pct_change'] = hist_ret['pct_change'].astype(float) / 100

    # 读取指数的收益率
    query = """
        select TRADE_DT, S_DQ_PCTCHANGE
        from AINDEXEODPRICES
        where TRADE_DT between '{hist}' and '{end}' and S_INFO_WINDCODE='{code}'
        order by TRADE_DT
    """.format(code=code, hist=hist, end=end)
    cursor.execute(query)
    data_mv = cursor.fetchall()
    port_mv = pd.DataFrame(data_mv, columns=['date', 'pct_change'])
    port_mv['pct_change'] = port_mv['pct_change'].astype(float) / 100
    port_mv.set_index('date', inplace=True)

    # 数据用parquet格式存储，用pd.read_parquet读入时可保留日期等数据的字符串格式
    hist_ret.to_parquet('hist returns(10-23).parquet', index=True)
    port_mv.to_parquet('CSI all index(10-23).parquet', index=True)


get_daily_codes('000985.CSI', '20100101', '20231231')
get_df('000985.CSI', '20100101', '20231231', 'factors(2010-2023).parquet')
get_industry('000985.CSI', '20100101', '20231231', 'industry(2010-2023).parquet')
get_returns('000985.CSI', '20100101', '20231231')
