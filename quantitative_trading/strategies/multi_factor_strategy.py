import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from ..factors.factor_calculator import FactorCalculator

class MultiFactorStrategy(BaseStrategy):
    def __init__(self, factors=None, weights=None):
        super().__init__()
        self.name = "Multi Factor Strategy"
        self.factor_calculator = FactorCalculator()
        self.factors = factors or ['momentum_20', 'reversal_20', 'volatility_20', 'volume_factor_20', 'macd']
        self.weights = weights or [0.2, 0.2, -0.1, 0.2, 0.3]  # 负号表示反向因子
    
    def calculate_factor_score(self, data):
        """计算因子得分"""
        factors = self.factor_calculator.calculate_all_factors(data)
        
        # 标准化因子
        normalized_factors = (factors - factors.mean()) / factors.std()
        
        # 计算综合得分
        score = np.dot(normalized_factors[self.factors], self.weights)
        return score
    
    def generate_signal(self, data):
        """基于因子得分生成信号"""
        score = self.calculate_factor_score(data)
        
        # 根据得分生成信号
        if score.iloc[-1] > 0.5:
            return 1  # 买入
        elif score.iloc[-1] < -0.5:
            return -1  # 卖出
        else:
            return 0  # 持有
    
    def backtest(self, data):
        """回测策略"""
        portfolio_values = []
        factor_scores = []
        
        for i in range(60, len(data)):  # 确保有足够数据计算因子
            window_data = data.iloc[:i+1]
            signal = self.generate_signal(window_data)
            price = data['close'].iloc[i]
            date = data.index[i]
            
            self.execute_trade(signal, price, date)
            self.update_portfolio(price)
            portfolio_values.append(self.portfolio_value)
            
            # 计算因子得分
            score = self.calculate_factor_score(window_data)
            factor_scores.append(score.iloc[-1])
        
        # 添加 portfolio 价值和因子得分到数据中
        data['portfolio_value'] = np.nan
        data.iloc[60:, data.columns.get_loc('portfolio_value')] = portfolio_values
        
        data['factor_score'] = np.nan
        data.iloc[60:, data.columns.get_loc('factor_score')] = factor_scores
        
        return data