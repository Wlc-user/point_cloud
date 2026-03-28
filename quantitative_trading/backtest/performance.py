import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceAnalyzer:
    def __init__(self):
        pass
    
    def calculate_returns(self, portfolio_values):
        """计算收益率"""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return returns
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """计算夏普比率"""
        annual_returns = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility
        return sharpe_ratio
    
    def calculate_max_drawdown(self, portfolio_values):
        """计算最大回撤"""
        cumulative_returns = portfolio_values / portfolio_values[0]
        peak = cumulative_returns[0]
        max_drawdown = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def calculate_calmar_ratio(self, returns, max_drawdown):
        """计算卡玛比率"""
        annual_returns = np.mean(returns) * 252
        if max_drawdown == 0:
            return np.inf
        return annual_returns / max_drawdown
    
    def calculate_win_rate(self, trades):
        """计算胜率"""
        if len(trades) == 0:
            return 0
        
        winning_trades = 0
        for i in range(1, len(trades)):
            if trades['signal'].iloc[i] == 'sell':
                buy_price = trades[trades['signal'] == 'buy']['price'].iloc[-1]
                sell_price = trades['price'].iloc[i]
                if sell_price > buy_price:
                    winning_trades += 1
        
        return winning_trades / len(trades[trades['signal'] == 'sell'])
    
    def calculate_profit_factor(self, trades):
        """计算盈亏比"""
        if len(trades) == 0:
            return 0
        
        total_profit = 0
        total_loss = 0
        
        for i in range(1, len(trades)):
            if trades['signal'].iloc[i] == 'sell':
                buy_price = trades[trades['signal'] == 'buy']['price'].iloc[-1]
                sell_price = trades['price'].iloc[i]
                profit = (sell_price - buy_price) * trades['shares'].iloc[i]
                if profit > 0:
                    total_profit += profit
                else:
                    total_loss += abs(profit)
        
        if total_loss == 0:
            return np.inf
        return total_profit / total_loss
    
    def analyze_performance(self, data, trades):
        """综合分析绩效"""
        portfolio_values = data['portfolio_value'].dropna().values
        returns = self.calculate_returns(portfolio_values)
        
        metrics = {
            'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
            'annual_return': np.mean(returns) * 252 * 100,
            'volatility': np.std(returns) * np.sqrt(252) * 100,
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values) * 100,
            'calmar_ratio': self.calculate_calmar_ratio(returns, self.calculate_max_drawdown(portfolio_values)),
            'win_rate': self.calculate_win_rate(trades) * 100,
            'profit_factor': self.calculate_profit_factor(trades),
            'total_trades': len(trades),
            'average_holding_period': self.calculate_average_holding_period(trades)
        }
        
        return metrics
    
    def calculate_average_holding_period(self, trades):
        """计算平均持有期"""
        if len(trades) < 2:
            return 0
        
        holding_periods = []
        buy_date = None
        
        for i, row in trades.iterrows():
            if row['signal'] == 'buy':
                buy_date = row['date']
            elif row['signal'] == 'sell' and buy_date:
                holding_period = (row['date'] - buy_date).days
                holding_periods.append(holding_period)
                buy_date = None
        
        if holding_periods:
            return np.mean(holding_periods)
        return 0
    
    def plot_equity_curve(self, data):
        """绘制权益曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['portfolio_value'], label='Strategy')
        plt.plot(data.index, data['close'] / data['close'].iloc[0] * 1000000, label='Benchmark')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_drawdown(self, data):
        """绘制回撤曲线"""
        portfolio_values = data['portfolio_value'].dropna()
        cumulative_returns = portfolio_values / portfolio_values.iloc[0]
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown * 100, label='Drawdown')
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def print_performance_report(self, metrics):
        """打印绩效报告"""
        print("=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:25}: {value:.2f}")
            else:
                print(f"{key:25}: {value}")
        print("=" * 60)