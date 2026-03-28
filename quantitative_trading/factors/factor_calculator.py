import pandas as pd
import numpy as np

class FactorCalculator:
    def __init__(self):
        pass
    
    def calculate_momentum(self, df, window=20):
        """计算动量因子"""
        return df['close'].pct_change(window)
    
    def calculate_reversal(self, df, window=20):
        """计算反转因子"""
        return -df['close'].pct_change(window)
    
    def calculate_volatility(self, df, window=20):
        """计算波动率因子"""
        return df['return'].rolling(window).std()
    
    def calculate_volume_factor(self, df, window=20):
        """计算成交量因子"""
        return df['vol'].rolling(window).mean() / df['vol'].rolling(120).mean()
    
    def calculate_price_volume_ratio(self, df, window=20):
        """计算量价因子"""
        price_change = df['close'].pct_change(window)
        volume_change = df['vol'].pct_change(window)
        return price_change / volume_change.replace(0, np.nan)
    
    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD因子"""
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return histogram
    
    def calculate_rsi(self, df, window=14):
        """计算RSI因子"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """计算布林带因子"""
        ma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return (df['close'] - lower) / (upper - lower)
    
    def calculate_kdj(self, df, window=9):
        """计算KDJ因子"""
        low = df['low'].rolling(window).min()
        high = df['high'].rolling(window).max()
        rsv = (df['close'] - low) / (high - low) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        return j
    
    def calculate_ma_crossover(self, df, short_window=5, long_window=20):
        """计算均线金叉因子"""
        short_ma = df['close'].rolling(short_window).mean()
        long_ma = df['close'].rolling(long_window).mean()
        return short_ma - long_ma
    
    def calculate_atr(self, df, window=14):
        """计算ATR因子"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        return atr / df['close']
    
    def calculate_all_factors(self, df):
        """计算所有因子"""
        factors = pd.DataFrame(index=df.index)
        factors['momentum_20'] = self.calculate_momentum(df, 20)
        factors['reversal_20'] = self.calculate_reversal(df, 20)
        factors['volatility_20'] = self.calculate_volatility(df, 20)
        factors['volume_factor_20'] = self.calculate_volume_factor(df, 20)
        factors['price_volume_ratio_20'] = self.calculate_price_volume_ratio(df, 20)
        factors['macd'] = self.calculate_macd(df)
        factors['rsi_14'] = self.calculate_rsi(df, 14)
        factors['bollinger_bands'] = self.calculate_bollinger_bands(df)
        factors['kdj'] = self.calculate_kdj(df)
        factors['ma_crossover'] = self.calculate_ma_crossover(df)
        factors['atr'] = self.calculate_atr(df)
        return factors