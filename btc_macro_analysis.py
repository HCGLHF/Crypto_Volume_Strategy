"""
BTC宏观回测分析 - Local Low与Volume异常检测
================================================
分析目标：验证BTC在local低点时volume反趋势增长是否预示反弹

策略逻辑：
1. Local Low定义：1个月内跌幅超过15%
2. Volume异常：该日volume的rolling median(前5天) > 该月其他volume的upper quantile
3. 反弹验证：未来30天内斜率超过1
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def fetch_btc_data(start_date='2018-01-01', end_date=None):
    """获取BTC日K线数据"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"正在获取BTC数据: {start_date} 至 {end_date}")
    btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    
    # 处理MultiIndex columns
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    btc = btc.reset_index()
    btc.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    btc['Date'] = pd.to_datetime(btc['Date'])
    btc = btc.sort_values('Date').reset_index(drop=True)
    
    print(f"成功获取 {len(btc)} 条数据记录")
    return btc


def find_local_lows(df, lookback_days=30, drop_threshold=0.15):
    """
    找到所有local low点
    定义：过去30天内跌幅超过15%的最低点
    """
    df = df.copy()
    df['rolling_max'] = df['Close'].rolling(window=lookback_days, min_periods=1).max()
    df['drawdown'] = (df['Close'] - df['rolling_max']) / df['rolling_max']
    
    # 找到跌幅超过阈值的点
    significant_drops = df[df['drawdown'] <= -drop_threshold].copy()
    
    local_lows = []
    
    if len(significant_drops) == 0:
        return pd.DataFrame()
    
    # 对于每个显著下跌区间，找到真正的局部最低点
    i = 0
    while i < len(significant_drops):
        # 找到连续的下跌区间
        start_idx = significant_drops.index[i]
        end_idx = start_idx
        
        j = i + 1
        while j < len(significant_drops):
            current_idx = significant_drops.index[j]
            prev_idx = significant_drops.index[j-1]
            
            # 如果两个点相隔不超过7天，认为是同一个下跌区间
            if (df.loc[current_idx, 'Date'] - df.loc[prev_idx, 'Date']).days <= 7:
                end_idx = current_idx
                j += 1
            else:
                break
        
        # 在这个区间内找到最低点
        segment = df.loc[start_idx:end_idx]
        min_idx = segment['Close'].idxmin()
        local_lows.append({
            'Date': df.loc[min_idx, 'Date'],
            'Close': df.loc[min_idx, 'Close'],
            'Volume': df.loc[min_idx, 'Volume'],
            'Drawdown': df.loc[min_idx, 'drawdown'],
            'Index': min_idx
        })
        
        i = j
    
    return pd.DataFrame(local_lows)


def check_volume_anomaly(df, local_low_idx, lookback_days=30, rolling_window=5, quantile=0.75):
    """
    检查volume异常
    条件：该日volume的rolling median(前5天) > 该月其他volume的upper quantile
    """
    if local_low_idx < lookback_days:
        return False, None, None
    
    # 获取该月的数据范围
    month_start = max(0, local_low_idx - lookback_days)
    month_data = df.iloc[month_start:local_low_idx + 1].copy()
    
    # 计算rolling median of volume (前5天)
    current_volume = df.iloc[local_low_idx]['Volume']
    
    # 获取前5天的volume计算median
    start_rolling = max(0, local_low_idx - rolling_window)
    rolling_volumes = df.iloc[start_rolling:local_low_idx + 1]['Volume']
    rolling_median = rolling_volumes.median()
    
    # 计算该月其他日期的volume upper quantile
    other_volumes = month_data[month_data.index != local_low_idx]['Volume']
    if len(other_volumes) == 0:
        return False, rolling_median, None
    
    upper_quantile = other_volumes.quantile(quantile)
    
    # 判断是否异常
    is_anomaly = rolling_median > upper_quantile
    
    return is_anomaly, rolling_median, upper_quantile


def check_rebound(df, local_low_idx, future_days=30, min_slope=1):
    """
    检查未来30天是否有反弹
    条件：30天内价格斜率超过1（每日涨幅百分比）
    """
    if local_low_idx + future_days >= len(df):
        return False, None, None, None
    
    future_data = df.iloc[local_low_idx:local_low_idx + future_days + 1].copy()
    
    low_price = future_data.iloc[0]['Close']
    max_price = future_data['Close'].max()
    max_price_idx = future_data['Close'].idxmax()
    
    days_to_max = max_price_idx - local_low_idx
    
    if days_to_max == 0:
        return False, 0, 0, days_to_max
    
    # 计算涨幅
    total_return = (max_price - low_price) / low_price * 100  # 百分比
    
    # 计算日均斜率（每日平均涨幅百分比）
    daily_slope = total_return / days_to_max
    
    is_rebound = daily_slope >= min_slope
    
    return is_rebound, total_return, daily_slope, days_to_max


def analyze_btc_local_lows(start_date='2018-01-01'):
    """主分析函数"""
    
    # 1. 获取数据
    df = fetch_btc_data(start_date)
    
    # 2. 找到local lows
    print("\n正在识别Local Low点...")
    local_lows = find_local_lows(df, lookback_days=30, drop_threshold=0.15)
    print(f"共找到 {len(local_lows)} 个Local Low点")
    
    if len(local_lows) == 0:
        print("未找到符合条件的Local Low点")
        return df, pd.DataFrame()
    
    # 3. 分析每个local low
    results = []
    
    for _, low in local_lows.iterrows():
        idx = low['Index']
        
        # 检查volume异常
        is_volume_anomaly, rolling_med, upper_q = check_volume_anomaly(df, idx)
        
        # 检查反弹
        is_rebound, total_return, daily_slope, days_to_max = check_rebound(df, idx)
        
        results.append({
            'Date': low['Date'],
            'Price': low['Close'],
            'Drawdown_%': low['Drawdown'] * 100,
            'Volume': low['Volume'],
            'Volume_Rolling_Median': rolling_med,
            'Volume_Upper_Quantile': upper_q,
            'Volume_Anomaly': is_volume_anomaly,
            'Has_Rebound': is_rebound,
            'Total_Return_%': total_return,
            'Daily_Slope_%': daily_slope,
            'Days_to_Max': days_to_max,
            'Index': idx
        })
    
    results_df = pd.DataFrame(results)
    
    # 4. 统计分析
    print("\n" + "="*60)
    print("分析结果统计")
    print("="*60)
    
    total_lows = len(results_df)
    volume_anomaly_count = results_df['Volume_Anomaly'].sum()
    rebound_count = results_df['Has_Rebound'].sum()
    
    print(f"\n总Local Low数量: {total_lows}")
    print(f"Volume异常数量: {volume_anomaly_count} ({volume_anomaly_count/total_lows*100:.1f}%)")
    print(f"成功反弹数量: {rebound_count} ({rebound_count/total_lows*100:.1f}%)")
    
    # 核心验证：Volume异常 + 反弹
    both_conditions = results_df[(results_df['Volume_Anomaly'] == True) & 
                                  (results_df['Has_Rebound'] == True)]
    
    # Volume异常但无反弹
    anomaly_no_rebound = results_df[(results_df['Volume_Anomaly'] == True) & 
                                     (results_df['Has_Rebound'] == False)]
    
    # 无Volume异常但有反弹
    no_anomaly_rebound = results_df[(results_df['Volume_Anomaly'] == False) & 
                                     (results_df['Has_Rebound'] == True)]
    
    # 无Volume异常且无反弹
    no_anomaly_no_rebound = results_df[(results_df['Volume_Anomaly'] == False) & 
                                        (results_df['Has_Rebound'] == False)]
    
    print(f"\n--- 策略验证 ---")
    print(f"Volume异常 + 反弹成功: {len(both_conditions)} 次")
    print(f"Volume异常 + 无反弹: {len(anomaly_no_rebound)} 次")
    print(f"无Volume异常 + 反弹成功: {len(no_anomaly_rebound)} 次")
    print(f"无Volume异常 + 无反弹: {len(no_anomaly_no_rebound)} 次")
    
    if volume_anomaly_count > 0:
        success_rate = len(both_conditions) / volume_anomaly_count * 100
        print(f"\n【关键指标】Volume异常时的反弹成功率: {success_rate:.1f}%")
    
    # 对比基准
    if total_lows > 0:
        base_rate = rebound_count / total_lows * 100
        print(f"【基准对比】所有Local Low的反弹率: {base_rate:.1f}%")
    
    # 详细结果表
    print("\n" + "="*60)
    print("详细分析结果")
    print("="*60)
    
    display_cols = ['Date', 'Price', 'Drawdown_%', 'Volume_Anomaly', 
                    'Has_Rebound', 'Total_Return_%', 'Daily_Slope_%']
    print(results_df[display_cols].to_string(index=False))
    
    return df, results_df


def create_visualization(df, results_df, save_html=True):
    """创建可视化K线图"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('BTC日K线图 - Local Low分析', 'Volume')
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='BTC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume柱状图
    colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' 
              for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 标记Local Low点
    if len(results_df) > 0:
        # Volume异常 + 反弹成功 (绿色大圆点)
        success = results_df[(results_df['Volume_Anomaly'] == True) & 
                            (results_df['Has_Rebound'] == True)]
        if len(success) > 0:
            fig.add_trace(
                go.Scatter(
                    x=success['Date'],
                    y=success['Price'],
                    mode='markers',
                    name='✓ Volume异常+反弹成功',
                    marker=dict(
                        size=18,
                        color='#00ff88',
                        symbol='star',
                        line=dict(color='#004d26', width=2)
                    ),
                    hovertemplate='<b>成功信号</b><br>日期: %{x}<br>价格: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Volume异常 + 无反弹 (红色X)
        fail = results_df[(results_df['Volume_Anomaly'] == True) & 
                         (results_df['Has_Rebound'] == False)]
        if len(fail) > 0:
            fig.add_trace(
                go.Scatter(
                    x=fail['Date'],
                    y=fail['Price'],
                    mode='markers',
                    name='✗ Volume异常+无反弹',
                    marker=dict(
                        size=14,
                        color='#ff4444',
                        symbol='x',
                        line=dict(color='#880000', width=2)
                    ),
                    hovertemplate='<b>失败信号</b><br>日期: %{x}<br>价格: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 普通Local Low (黄色三角)
        normal = results_df[results_df['Volume_Anomaly'] == False]
        if len(normal) > 0:
            fig.add_trace(
                go.Scatter(
                    x=normal['Date'],
                    y=normal['Price'],
                    mode='markers',
                    name='○ 普通Local Low',
                    marker=dict(
                        size=10,
                        color='#ffaa00',
                        symbol='triangle-up',
                        line=dict(color='#664400', width=1)
                    ),
                    hovertemplate='<b>普通低点</b><br>日期: %{x}<br>价格: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # 布局设置
    fig.update_layout(
        title={
            'text': 'BTC宏观回测分析 - Local Low与Volume异常检测',
            'x': 0.5,
            'font': dict(size=20, color='#e0e0e0')
        },
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=900,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # 坐标轴样式
    fig.update_xaxes(
        gridcolor='#1f2937',
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor='#1f2937',
        showgrid=True,
        zeroline=False,
        tickformat='$,.0f',
        row=1, col=1
    )
    fig.update_yaxes(
        gridcolor='#1f2937',
        showgrid=True,
        zeroline=False,
        row=2, col=1
    )
    
    if save_html:
        output_file = 'btc_local_low_analysis.html'
        fig.write_html(output_file)
        print(f"\n图表已保存至: {output_file}")
    
    fig.show()
    
    return fig


def main():
    """主入口"""
    print("="*60)
    print("BTC宏观回测分析 - Local Low与Volume异常检测")
    print("="*60)
    print("\n策略假设：BTC在local低点时volume反趋势增长可能带动涨幅")
    print("\n参数设置：")
    print("  - Local Low定义：1个月内跌幅 > 15%")
    print("  - Volume异常：Rolling Median(5天) > 月度Upper Quantile(75%)")
    print("  - 反弹确认：未来30天内日均斜率 > 1%")
    print("-"*60)
    
    # 执行分析
    df, results_df = analyze_btc_local_lows(start_date='2018-01-01')
    
    # 创建可视化
    if len(results_df) > 0:
        create_visualization(df, results_df)
    
    # 保存结果
    if len(results_df) > 0:
        results_df.to_csv('btc_analysis_results.csv', index=False)
        print(f"\n分析结果已保存至: btc_analysis_results.csv")
    
    return df, results_df


if __name__ == '__main__':
    df, results = main()
