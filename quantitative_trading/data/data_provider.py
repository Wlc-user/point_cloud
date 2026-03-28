import pandas as pd
import numpy as np
import tushare as ts
import time
from datetime import datetime, timedelta

class DataProvider:
    def __init__(self, token=None):
        if token:
            ts.set_token(token)
        self.pro = ts.pro_api()
        self.data_cache = {}
    
    def get_stock_list(self, market='SZSE'):
        """获取股票列表"""
        df = self.pro.stock_basic(exchange=market, list_status='L')
        return df
    
    def get_daily_data(self, ts_code, start_date, end_date):
        """获取日线数据"""
        cache_key = f"daily_{ts_code}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        # 计算基本指标
        df['return'] = df['close'].pct_change()
        df['volume_change'] = df['vol'].pct_change()
        
        self.data_cache[cache_key] = df
        return df
    
    def get_batch_daily_data(self, ts_codes, start_date, end_date):
        """批量获取多只股票的日线数据"""
        data_dict = {}
        for ts_code in ts_codes:
            try:
                data = self.get_daily_data(ts_code, start_date, end_date)
                data_dict[ts_code] = data
                time.sleep(0.1)  # 避免API限流
            except Exception as e:
                print(f"获取{ts_code}数据失败: {e}")
        return data_dict
    
    def get_basic_financial(self, ts_code, start_date, end_date):
        """获取基本面数据"""
        df = self.pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('end_date')
        df['end_date'] = pd.to_datetime(df['end_date'])
        df.set_index('end_date', inplace=True)
        return df
    
    def get_index_data(self, ts_code, start_date, end_date):
        """获取指数数据"""
        df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df['return'] = df['close'].pct_change()
        return df
    
    def get_future_data(self, ts_code, start_date, end_date):
        """获取期货数据"""
        df = self.pro.fut_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df['return'] = df['close'].pct_change()
        return df