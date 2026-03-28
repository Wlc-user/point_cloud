# 量化交易系统

这是一个基于Python的量化交易系统，包含数据获取、因子计算、策略回测和绩效评估等功能。

## 功能特性

- **数据获取**：通过tushare API获取A股市场数据
- **因子计算**：实现了多种常用量化因子
- **策略回测**：支持双均线策略和多因子策略
- **绩效评估**：计算夏普比率、最大回撤等关键指标
- **策略比较**：对比不同策略的表现

## 项目结构

```
quantitative_trading/
├── data/
│   └── data_provider.py        # 数据获取模块
├── factors/
│   └── factor_calculator.py    # 因子计算模块
├── strategies/
│   ├── base_strategy.py        # 基础策略类
│   ├── ma_strategy.py          # 双均线策略
│   └── multi_factor_strategy.py # 多因子策略
├── backtest/
│   └── performance.py          # 绩效评估模块
├── main.py                     # 主脚本
└── requirements.txt            # 依赖文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. **获取tushare API Token**
   - 注册tushare账号：https://tushare.pro/register
   - 获取API Token：https://tushare.pro/user/token

2. **修改main.py中的token**
   ```python
   token = "your_tushare_token_here"  # 替换为你的token
   ```

3. **运行策略**
   ```bash
   python main.py
   ```

## 策略说明

### 双均线策略 (MA Strategy)
- 基于短期均线和长期均线的交叉信号
- 参数：short_window=5, long_window=20
- 信号生成：短期均线上穿长期均线买入，下穿卖出

### 多因子策略 (Multi Factor Strategy)
- 基于多个因子的综合得分
- 因子包括：动量、反转、波动率、成交量、MACD等
- 信号生成：根据因子得分阈值生成买入/卖出信号

## 绩效指标

- **总收益率**：策略的整体收益
- **年化收益率**：年度化的投资回报
- **波动率**：策略收益的标准差
- **夏普比率**：每单位风险的超额收益
- **最大回撤**：从峰值到谷值的最大损失
- **卡玛比率**：年化收益率与最大回撤的比值
- **胜率**：盈利交易占比
- **盈亏比**：平均盈利与平均亏损的比率
- **总交易次数**：策略执行的交易次数
- **平均持有期**：平均持仓天数

## 示例输出

运行main.py后，系统会：
1. 获取指定股票的历史数据
2. 运行双均线策略和多因子策略
3. 计算并打印绩效报告
4. 绘制权益曲线和回撤曲线
5. 比较两个策略的表现

## 扩展建议

1. **添加更多策略**：如趋势跟踪、均值回归、套利策略等
2. **优化因子**：添加更多因子，如基本面因子、另类数据因子等
3. **机器学习**：使用机器学习模型预测股票走势
4. **实盘交易**：对接券商API实现自动交易
5. **风险管理**：添加风险控制模块

## 注意事项

- 本系统仅用于回测和研究，不构成投资建议
- 实盘交易需考虑交易成本、滑点等因素
- 历史表现不代表未来收益
- tushare API有调用频率限制，请注意合理使用

## 许可证

MIT License