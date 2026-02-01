# -*- coding: utf-8 -*-
"""
BTC宏观回测分析 - 交互式Web应用
================================
功能：
- 可调整所有分析参数
- Run按钮重新推理
- 数据下载功能
- 重新获取数据按钮
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="BTC Local Low Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7931a, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #3d3d5c;
    }
    .success-metric {
        color: #00ff88;
        font-size: 2rem;
        font-weight: bold;
    }
    .fail-metric {
        color: #ff4444;
        font-size: 2rem;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .sidebar .stButton > button {
        background: linear-gradient(90deg, #f7931a, #ff6b35);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 数据获取函数 ====================
@st.cache_data(ttl=3600)  # 缓存1小时
def fetch_btc_data(start_date, end_date=None):
    """获取BTC日K线数据"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    btc = btc.reset_index()
    btc.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    btc['Date'] = pd.to_datetime(btc['Date'])
    btc = btc.sort_values('Date').reset_index(drop=True)
    
    return btc


def force_fetch_btc_data(start_date, end_date=None):
    """强制重新获取数据（清除缓存）"""
    fetch_btc_data.clear()
    return fetch_btc_data(start_date, end_date)


# ==================== 分析函数 ====================
def find_local_lows(df, lookback_days, drop_threshold, merge_days):
    """找到所有local low点"""
    df = df.copy()
    df['rolling_max'] = df['Close'].rolling(window=lookback_days, min_periods=1).max()
    df['drawdown'] = (df['Close'] - df['rolling_max']) / df['rolling_max']
    
    significant_drops = df[df['drawdown'] <= -drop_threshold].copy()
    local_lows = []
    
    if len(significant_drops) == 0:
        return pd.DataFrame()
    
    i = 0
    while i < len(significant_drops):
        start_idx = significant_drops.index[i]
        end_idx = start_idx
        
        j = i + 1
        while j < len(significant_drops):
            current_idx = significant_drops.index[j]
            prev_idx = significant_drops.index[j-1]
            
            if (df.loc[current_idx, 'Date'] - df.loc[prev_idx, 'Date']).days <= merge_days:
                end_idx = current_idx
                j += 1
            else:
                break
        
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


def check_volume_anomaly(df, local_low_idx, lookback_days, rolling_window, quantile):
    """检查volume异常"""
    if local_low_idx < lookback_days:
        return False, None, None
    
    month_start = max(0, local_low_idx - lookback_days)
    month_data = df.iloc[month_start:local_low_idx + 1].copy()
    
    start_rolling = max(0, local_low_idx - rolling_window)
    rolling_volumes = df.iloc[start_rolling:local_low_idx + 1]['Volume']
    rolling_median = rolling_volumes.median()
    
    other_volumes = month_data[month_data.index != local_low_idx]['Volume']
    if len(other_volumes) == 0:
        return False, rolling_median, None
    
    upper_quantile = other_volumes.quantile(quantile)
    is_anomaly = rolling_median > upper_quantile
    
    return is_anomaly, rolling_median, upper_quantile


def check_rebound(df, local_low_idx, future_days, min_slope):
    """检查反弹"""
    if local_low_idx + future_days >= len(df):
        return False, None, None, None
    
    future_data = df.iloc[local_low_idx:local_low_idx + future_days + 1].copy()
    
    low_price = future_data.iloc[0]['Close']
    max_price = future_data['Close'].max()
    max_price_idx = future_data['Close'].idxmax()
    
    days_to_max = max_price_idx - local_low_idx
    
    if days_to_max == 0:
        return False, 0, 0, days_to_max
    
    total_return = (max_price - low_price) / low_price * 100
    daily_slope = total_return / days_to_max
    is_rebound = daily_slope >= min_slope
    
    return is_rebound, total_return, daily_slope, days_to_max


def run_analysis(df, params):
    """运行完整分析"""
    local_lows = find_local_lows(
        df, 
        params['lookback_days'], 
        params['drop_threshold'],
        params['merge_days']
    )
    
    if len(local_lows) == 0:
        return pd.DataFrame()
    
    results = []
    
    for _, low in local_lows.iterrows():
        idx = low['Index']
        
        is_volume_anomaly, rolling_med, upper_q = check_volume_anomaly(
            df, idx, 
            params['lookback_days'], 
            params['rolling_window'], 
            params['volume_quantile']
        )
        
        is_rebound, total_return, daily_slope, days_to_max = check_rebound(
            df, idx, 
            params['future_days'], 
            params['min_slope']
        )
        
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
    
    return pd.DataFrame(results)


# ==================== 可视化函数 ====================
def create_chart(df, results_df):
    """创建K线图"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('BTC Daily K-Line - Local Low Analysis', 'Volume')
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
    
    # 标记点
    if len(results_df) > 0:
        # 成功信号
        success = results_df[(results_df['Volume_Anomaly'] == True) & 
                            (results_df['Has_Rebound'] == True)]
        if len(success) > 0:
            fig.add_trace(
                go.Scatter(
                    x=success['Date'],
                    y=success['Price'],
                    mode='markers',
                    name='✓ Vol Anomaly + Rebound',
                    marker=dict(size=18, color='#00ff88', symbol='star',
                               line=dict(color='#004d26', width=2)),
                    hovertemplate='<b>SUCCESS</b><br>Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 失败信号
        fail = results_df[(results_df['Volume_Anomaly'] == True) & 
                         (results_df['Has_Rebound'] == False)]
        if len(fail) > 0:
            fig.add_trace(
                go.Scatter(
                    x=fail['Date'],
                    y=fail['Price'],
                    mode='markers',
                    name='✗ Vol Anomaly + No Rebound',
                    marker=dict(size=14, color='#ff4444', symbol='x',
                               line=dict(color='#880000', width=2)),
                    hovertemplate='<b>FAILED</b><br>Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 普通低点
        normal = results_df[results_df['Volume_Anomaly'] == False]
        if len(normal) > 0:
            fig.add_trace(
                go.Scatter(
                    x=normal['Date'],
                    y=normal['Price'],
                    mode='markers',
                    name='○ Normal Local Low',
                    marker=dict(size=10, color='#ffaa00', symbol='triangle-up',
                               line=dict(color='#664400', width=1)),
                    hovertemplate='<b>NORMAL</b><br>Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=700,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(gridcolor='#1f2937', showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor='#1f2937', showgrid=True, zeroline=False, 
                     tickformat='$,.0f', row=1, col=1)
    fig.update_yaxes(gridcolor='#1f2937', showgrid=True, zeroline=False, row=2, col=1)
    
    return fig


# ==================== 主界面 ====================
def main():
    # 标题
    st.markdown('<h1 class="main-header">₿ BTC Local Low & Volume Anomaly Analysis</h1>', 
                unsafe_allow_html=True)
    
    # 侧边栏 - 参数设置
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        st.subheader("📅 Data Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2018, 1, 1),
                min_value=datetime(2014, 1, 1),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                min_value=datetime(2014, 1, 1),
                max_value=datetime.now()
            )
        
        st.divider()
        
        st.subheader("📉 Local Low Detection")
        lookback_days = st.slider(
            "Lookback Days (回看天数)",
            min_value=7, max_value=90, value=30,
            help="用于计算最高点的回看天数"
        )
        
        drop_threshold = st.slider(
            "Drop Threshold % (跌幅阈值)",
            min_value=5.0, max_value=50.0, value=15.0, step=1.0,
            help="定义Local Low的最小跌幅百分比"
        ) / 100
        
        merge_days = st.slider(
            "Merge Days (合并天数)",
            min_value=1, max_value=30, value=7,
            help="相隔多少天内的低点视为同一个下跌区间"
        )
        
        st.divider()
        
        st.subheader("📊 Volume Anomaly")
        rolling_window = st.slider(
            "Rolling Window (滚动窗口)",
            min_value=2, max_value=20, value=5,
            help="计算Volume Rolling Median的窗口大小"
        )
        
        volume_quantile = st.slider(
            "Volume Quantile (分位数)",
            min_value=0.5, max_value=0.99, value=0.75, step=0.05,
            help="用于比较的Volume分位数阈值"
        )
        
        st.divider()
        
        st.subheader("🚀 Rebound Confirmation")
        future_days = st.slider(
            "Future Days (未来天数)",
            min_value=7, max_value=90, value=30,
            help="检查反弹的时间窗口"
        )
        
        min_slope = st.slider(
            "Min Daily Slope % (最小日斜率)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="确认反弹的最小日均涨幅百分比"
        )
        
        st.divider()
        
        # 按钮区域
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_button = st.button("▶️ Run Analysis", type="primary", use_container_width=True)
        with col_btn2:
            refresh_button = st.button("🔄 Refresh Data", use_container_width=True)
    
    # 参数字典
    params = {
        'lookback_days': lookback_days,
        'drop_threshold': drop_threshold,
        'merge_days': merge_days,
        'rolling_window': rolling_window,
        'volume_quantile': volume_quantile,
        'future_days': future_days,
        'min_slope': min_slope
    }
    
    # 初始化session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    
    # 刷新数据按钮
    if refresh_button:
        with st.spinner("🔄 Fetching fresh BTC data..."):
            st.session_state.df = force_fetch_btc_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            st.session_state.last_fetch_time = datetime.now()
            st.session_state.results_df = None
        st.success(f"✅ Data refreshed! {len(st.session_state.df)} records loaded.")
        st.rerun()
    
    # 运行分析按钮
    if run_button or st.session_state.df is None:
        with st.spinner("📊 Running analysis..."):
            # 获取数据
            if st.session_state.df is None or refresh_button:
                st.session_state.df = fetch_btc_data(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                st.session_state.last_fetch_time = datetime.now()
            
            # 运行分析
            st.session_state.results_df = run_analysis(st.session_state.df, params)
    
    # 显示结果
    if st.session_state.df is not None and st.session_state.results_df is not None:
        df = st.session_state.df
        results_df = st.session_state.results_df
        
        # 数据信息
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.info(f"📅 Data Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        with col_info2:
            st.info(f"📊 Total Records: {len(df)}")
        with col_info3:
            if st.session_state.last_fetch_time:
                st.info(f"🕐 Last Fetch: {st.session_state.last_fetch_time.strftime('%H:%M:%S')}")
        
        # 统计指标
        if len(results_df) > 0:
            st.subheader("📈 Analysis Summary")
            
            total_lows = len(results_df)
            volume_anomaly_count = results_df['Volume_Anomaly'].sum()
            rebound_count = results_df['Has_Rebound'].sum()
            
            both_conditions = results_df[(results_df['Volume_Anomaly'] == True) & 
                                          (results_df['Has_Rebound'] == True)]
            anomaly_no_rebound = results_df[(results_df['Volume_Anomaly'] == True) & 
                                             (results_df['Has_Rebound'] == False)]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Local Lows", total_lows)
            with col2:
                st.metric("Volume Anomalies", f"{volume_anomaly_count} ({volume_anomaly_count/total_lows*100:.1f}%)")
            with col3:
                st.metric("Successful Rebounds", f"{rebound_count} ({rebound_count/total_lows*100:.1f}%)")
            with col4:
                if volume_anomaly_count > 0:
                    success_rate = len(both_conditions) / volume_anomaly_count * 100
                    baseline = rebound_count / total_lows * 100
                    delta = success_rate - baseline
                    st.metric(
                        "Vol Anomaly Success Rate", 
                        f"{success_rate:.1f}%",
                        delta=f"{delta:.1f}% vs baseline",
                        delta_color="normal" if delta > 0 else "inverse"
                    )
                else:
                    st.metric("Vol Anomaly Success Rate", "N/A")
            
            # 详细统计
            st.subheader("📋 Strategy Validation Matrix")
            col_matrix1, col_matrix2 = st.columns(2)
            
            with col_matrix1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1a4d1a 0%, #2d5a2d 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="color: #88ff88; font-size: 0.9rem;">Vol Anomaly + Rebound ✓</div>
                    <div style="color: #00ff88; font-size: 2.5rem; font-weight: bold;">{}</div>
                </div>
                """.format(len(both_conditions)), unsafe_allow_html=True)
            
            with col_matrix2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4d1a1a 0%, #5a2d2d 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="color: #ff8888; font-size: 0.9rem;">Vol Anomaly + No Rebound ✗</div>
                    <div style="color: #ff4444; font-size: 2.5rem; font-weight: bold;">{}</div>
                </div>
                """.format(len(anomaly_no_rebound)), unsafe_allow_html=True)
            
            # 图表
            st.subheader("📊 Interactive Chart")
            fig = create_chart(df, results_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # 详细数据表
            st.subheader("📋 Detailed Results")
            
            display_df = results_df.copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.2f}")
            display_df['Drawdown_%'] = display_df['Drawdown_%'].apply(lambda x: f"{x:.2f}%")
            display_df['Total_Return_%'] = display_df['Total_Return_%'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )
            display_df['Daily_Slope_%'] = display_df['Daily_Slope_%'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )
            
            st.dataframe(
                display_df[['Date', 'Price', 'Drawdown_%', 'Volume_Anomaly', 
                           'Has_Rebound', 'Total_Return_%', 'Daily_Slope_%']],
                use_container_width=True,
                hide_index=True
            )
            
            # 下载区域
            st.subheader("💾 Download Data")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                # 下载原始BTC数据
                csv_raw = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Raw BTC Data (CSV)",
                    data=csv_raw,
                    file_name=f"btc_raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_dl2:
                # 下载分析结果
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Results (CSV)",
                    data=csv_results,
                    file_name=f"btc_analysis_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_dl3:
                # 下载图表HTML
                html_buffer = io.StringIO()
                fig.write_html(html_buffer)
                st.download_button(
                    label="📥 Download Chart (HTML)",
                    data=html_buffer.getvalue(),
                    file_name=f"btc_chart_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    use_container_width=True
                )
        
        else:
            st.warning("⚠️ No Local Low points found with current parameters. Try adjusting the thresholds.")
    
    # 参数说明
    with st.expander("📖 Parameter Guide"):
        st.markdown("""
        ### Local Low Detection
        - **Lookback Days**: 用于计算过去N天内的最高价，以此来计算回撤幅度
        - **Drop Threshold %**: 只有回撤超过此阈值的点才被视为Local Low
        - **Merge Days**: 如果两个低点相隔不超过N天，则视为同一个下跌区间
        
        ### Volume Anomaly Detection
        - **Rolling Window**: 计算当前日期前N天的Volume中位数
        - **Volume Quantile**: 与当月其他日期Volume的分位数进行比较（如0.75表示75%分位）
        
        ### Rebound Confirmation
        - **Future Days**: 检查低点后N天内是否出现反弹
        - **Min Daily Slope %**: 反弹确认的最小日均涨幅（总涨幅/达到最高点的天数）
        
        ### Strategy Logic
        如果在Local Low时，Volume的Rolling Median > 该月Volume的Upper Quantile，
        则认为出现了Volume异常（反趋势增长），观察未来是否有显著反弹。
        """)


if __name__ == '__main__':
    main()
