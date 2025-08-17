import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import kaleido
import os

from plotly.subplots import make_subplots

def parabola(x, a, b, c):
    """Define a parabolic curve (quadratic fit)"""
    return a * x**2 + b * x + c

# Function to plot and crop the pattern
def plot_cup_and_handle_pattern(data, complete_pattern):
    """
    Plot cup and handle pattern only (no breakout analysis)
    
    Args:
        data: Full dataset
        complete_pattern: Complete pattern dictionary
    """
    
    pattern_id = complete_pattern['pattern_id']
    cup_pattern = complete_pattern['cup']
    handle_pattern = complete_pattern['handle']
    
    # Extract data sections
    cup_start_idx = cup_pattern['cup_start_idx']
    cup_end_idx = cup_pattern['cup_end_idx']
    handle_start_idx = handle_pattern['handle_start_idx']
    handle_end_idx = handle_pattern['handle_end_idx']
    
    cup_data = data.iloc[cup_start_idx:cup_end_idx]
    handle_data = data.iloc[handle_start_idx:handle_end_idx]
    
    # Calculate pattern length for centering
    pattern_length = handle_end_idx - cup_start_idx
    post_handle_candles = 20  # Show 20 candles after handle ends
    
    # Center the pattern by using balanced context
    context_padding = 20  # At least 20 candles
    
    context_start_idx = max(0, cup_start_idx - context_padding)
    context_end_idx = min(len(data), handle_end_idx + post_handle_candles + context_padding)
    context_data = data.iloc[context_start_idx:context_end_idx]
    
    # Get post-handle data (20 candles after handle ends) for additional visualization
    post_handle_start_idx = handle_end_idx
    post_handle_end_idx = min(len(data), handle_end_idx + post_handle_candles)
    post_handle_data = data.iloc[post_handle_start_idx:post_handle_end_idx] if post_handle_end_idx > post_handle_start_idx else pd.DataFrame()
    
    # Create plot with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.75, 0.25]
    )
    
    # Background context
    fig.add_trace(go.Candlestick(
        x=context_data['open_time'],
        open=context_data['open'],
        high=context_data['high'],
        low=context_data['low'],
        close=context_data['close'],
        name='Context',
        increasing_line_color='darkgreen',
        decreasing_line_color='darkred',
        opacity=0.3
    ), row=1, col=1)
    
    # Cup pattern (highlighted)
    fig.add_trace(go.Candlestick(
        x=cup_data['open_time'],
        open=cup_data['open'],
        high=cup_data['high'],
        low=cup_data['low'],
        close=cup_data['close'],
        name='Cup',
        increasing_line_color='lime',
        decreasing_line_color='red'
    ), row=1, col=1)
    
    # Handle pattern (highlighted)
    fig.add_trace(go.Candlestick(
        x=handle_data['open_time'],
        open=handle_data['open'],
        high=handle_data['high'],
        low=handle_data['low'],
        close=handle_data['close'],
        name='Handle',
        increasing_line_color='yellow',
        decreasing_line_color='orange'
    ), row=1, col=1)
    
    # Post-handle data (20 candles after handle ends - shows breakout and post-breakout)
    if not post_handle_data.empty:
        fig.add_trace(go.Candlestick(
            x=post_handle_data['open_time'],
            open=post_handle_data['open'],
            high=post_handle_data['high'],
            low=post_handle_data['low'],
            close=post_handle_data['close'],
            name='Breakout Zone',
            increasing_line_color='lightgreen',
            decreasing_line_color='lightcoral'
        ), row=1, col=1)

    # Parabolic fit for cup
    if len(cup_pattern['parabola_params']) == 3:
        x_norm = np.arange(len(cup_data)) / (len(cup_data) - 1)
        y_data = cup_data['smooth_close'].values
        y_min, y_max = y_data.min(), y_data.max()
        y_range = y_max - y_min
        
        fitted_norm = parabola(x_norm, *cup_pattern['parabola_params'])
        fitted = fitted_norm * y_range + y_min
        
        fig.add_trace(go.Scatter(
            x=cup_data['open_time'],
            y=fitted,
            mode='lines',
            name=f'Parabolic Fit',
            line=dict(color='white', width=6, dash='solid')
        ), row=1, col=1)
    
    # Pattern boundary markers
    cup_start_time = cup_data.iloc[0]['open_time']
    handle_start_time = handle_data.iloc[0]['open_time']
    handle_end_time = handle_data.iloc[-1]['open_time']
    
    fig.add_vline(x=cup_start_time, line=dict(color='blue', width=2, dash='solid'))
    fig.add_vline(x=handle_start_time, line=dict(color='purple', width=2, dash='solid'))
    fig.add_vline(x=handle_end_time, line=dict(color='orange', width=2, dash='solid'))
    
    # Handle resistance line
    handle_resistance = handle_pattern['handle_high']
    fig.add_hline(y=handle_resistance, 
                  line=dict(color='red', width=2, dash='dash'))
    
    # Volume analysis
    combined_data = pd.concat([cup_data, handle_data])
    if 'volume' in combined_data.columns:
        fig.add_trace(go.Bar(
            x=combined_data['open_time'],
            y=combined_data['volume'],
            name='Volume',
            marker_color='gray',
            opacity=0.6
        ), row=2, col=1)
    
    # Layout and styling
    start_time_readable = pd.to_datetime(cup_start_time, unit='ms').strftime('%Y-%m-%d %H:%M')
    end_time_readable = pd.to_datetime(handle_end_time, unit='ms').strftime('%Y-%m-%d %H:%M')
    
    fig.update_layout(
        title=f"Cup & Handle Pattern #{pattern_id}<br>",
        template="plotly_dark",
        width=1600,
        height=900,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        )
    )
    
    # Update axes with full date and time
    fig.update_xaxes(
        type='date', 
        tickformat='%Y-%m-%d<br>%H:%M',  # Show date on top line, time on bottom line
        rangeslider=dict(visible=False),
        tickangle=-45,  # Rotate labels for better readability
        title_text="Date & Time"
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Save plot
    os.makedirs("patterns", exist_ok=True)
    filename = f'patterns/cup_handle_{pattern_id}.png'
    fig.write_image(filename, scale=2)