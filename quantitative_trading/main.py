import pandas as pd
import numpy as np
from data.data_provider import DataProvider
from factors.factor_calculator import FactorCalculator
from strategies.ma_strategy import MAStrategy
from strategies.multi_factor_strategy import MultiFactorStrategy
from backtest.performance import PerformanceAnalyzer

class QuantitativeTradingSystem:
    def __init__(self, token=None):
        self.data_provider = DataProvider(token)
        self.factor_calculator = FactorCalculator()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def run_ma_strategy(self, ts_code, start_date, end_date):
        """运行双均线策略"""
        print(f"Running MA Strategy for {ts_code}")
        
        # 获取数据
        data = self.data_provider.get_daily_data(ts_code, start_date, end_date)
        
        # 初始化策略
        strategy = MAStrategy(short_window=5, long_window=20)
        
        # 回测
        backtest_result = strategy.backtest(data)
        
        # 获取交易记录
        trades = strategy.get_trades()
        
        # 分析绩效
        metrics = self.performance_analyzer.analyze_performance(backtest_result, trades)
        
        # 打印报告
        self.performance_analyzer.print_performance_report(metrics)
        
        # 绘制图表
        self.performance_analyzer.plot_equity_curve(backtest_result)
        self.performance_analyzer.plot_drawdown(backtest_result)
        
        return backtest_result, metrics
    
    def run_multi_factor_strategy(self, ts_code, start_date, end_date):
        """运行多因子策略"""
        print(f"Running Multi Factor Strategy for {ts_code}")
        
        # 获取数据
        data = self.data_provider.get_daily_data(ts_code, start_date, end_date)
        
        # 初始化策略
        strategy = MultiFactorStrategy()
        
        # 回测
        backtest_result = strategy.backtest(data)
        
        # 获取交易记录
        trades = strategy.get_trades()
        
        # 分析绩效
        metrics = self.performance_analyzer.analyze_performance(backtest_result, trades)
        
        # 打印报告
        self.performance_analyzer.print_performance_report(metrics)
        
        # 绘制图表
        self.performance_analyzer.plot_equity_curve(backtest_result)
        self.performance_analyzer.plot_drawdown(backtest_result)
        
        return backtest_result, metrics
    
    def compare_strategies(self, ts_code, start_date, end_date):
        """比较不同策略"""
        print(f"Comparing strategies for {ts_code}")
        
        # 运行双均线策略
        ma_result, ma_metrics = self.run_ma_strategy(ts_code, start_date, end_date)
        
        # 运行多因子策略
        mf_result, mf_metrics = self.run_multi_factor_strategy(ts_code, start_date, end_date)
        
        # 比较结果
        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)
        print(f"{'Metric':25} {'MA Strategy':15} {'Multi Factor':15}")
        print("-" * 60)
        
        for key in ma_metrics:
            if key in mf_metrics:
                ma_value = ma_metrics[key]
                mf_value = mf_metrics[key]
                if isinstance(ma_value, float):
                    print(f"{key:25} {ma_value:15.2f} {mf_value:15.2f}")
                else:
                    print(f"{key:25} {str(ma_value):15} {str(mf_value):15}")
        
        print("=" * 60)
        
        return ma_result, ma_metrics, mf_result, mf_metrics

if __name__ == "__main__":
    # 初始化系统
    # 注意：需要替换为你自己的tushare token
    token = "your_tushare_token_here"
    qts = QuantitativeTradingSystem(token)
    
    # 运行策略
    ts_code = "000001.SZ"  # 平安银行
    start_date = "20200101"
    end_date = "20231231"
    
    # 比较策略
    qts.compare_strategies(ts_code, start_date, end_date)