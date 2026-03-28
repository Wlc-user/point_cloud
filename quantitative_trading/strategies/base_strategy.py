import pandas as pd
import numpy as np

class BaseStrategy:
    def __init__(self):
        self.name = "Base Strategy"
        self.position = 0
        self.capital = 1000000
        self.portfolio_value = 1000000
        self.trades = []
    
    def generate_signal(self, data):
        """生成交易信号"""
        raise NotImplementedError
    
    def execute_trade(self, signal, price, date):
        """执行交易"""
        if signal == 1 and self.position <= 0:
            # 买入
            shares = int(self.capital * 0.9 / price)  # 使用90%资金
            cost = shares * price
            self.position = shares
            self.capital -= cost
            self.trades.append({
                'date': date,
                'signal': 'buy',
                'price': price,
                'shares': shares,
                'cost': cost
            })
        elif signal == -1 and self.position >= 0:
            # 卖出
            proceeds = self.position * price
            self.capital += proceeds
            self.trades.append({
                'date': date,
                'signal': 'sell',
                'price': price,
                'shares': self.position,
                'proceeds': proceeds
            })
            self.position = 0
    
    def update_portfolio(self, price):
        """更新 portfolio 价值"""
        market_value = self.position * price
        self.portfolio_value = self.capital + market_value
    
    def get_trades(self):
        """获取交易记录"""
        return pd.DataFrame(self.trades)
    
    def get_portfolio_value(self):
        """获取 portfolio 价值"""
        return self.portfolio_value