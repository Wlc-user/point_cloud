import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MAStrategy(BaseStrategy):
    def __init__(self, short_window=5, long_window=20):
        super().__init__()
        self.name = "MA Strategy"
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, data):
        """基于双均线交叉生成信号"""
        # 计算移动平均线
        data['short_ma'] = data['close'].rolling(self.short_window).mean()
        data['long_ma'] = data['close'].rolling(self.long_window).mean()
        
        # 生成信号
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
        
        # 避免重复信号
        data['signal'] = data['signal'] * data['signal'].shift().fillna(0)
        data['signal'] = data['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return data['signal'].iloc[-1]
    
    def backtest(self, data):
        """回测策略"""
        portfolio_values = []
        
        for i in range(max(self.short_window, self.long_window), len(data)):
            window_data = data.iloc[:i+1]
            signal = self.generate_signal(window_data)
            price = data['close'].iloc[i]
            date = data.index[i]
            
            self.execute_trade(signal, price, date)
            self.update_portfolio(price)
            portfolio_values.append(self.portfolio_value)
        
        # 添加 portfolio 价值到数据中
        data['portfolio_value'] = np.nan
        data.iloc[max(self.short_window, self.long_window):, data.columns.get_loc('portfolio_value')] = portfolio_values
        
        return data