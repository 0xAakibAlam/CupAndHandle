import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import kaleido
import os

# Import parabola functions from pattern_detector
from pattern_detector import parabola, fit_parabola_to_cup

# Function to plot and crop the pattern
def plot_and_crop_pattern(cup_candidate, handle_candidate, breakout_time, pattern_id, full_data, pattern_start_idx, pattern_end_idx):
    # Add context: show 50 candles before and after the pattern for better visualization
    context_before = 50
    context_after = 50
    
    start_context_idx = max(0, pattern_start_idx - context_before)
    end_context_idx = min(len(full_data), pattern_end_idx + context_after)
    
    # Get the context data for background
    context_data = full_data.iloc[start_context_idx:end_context_idx]
    
    # Fit a parabola to the cup for the smooth arc using the improved function
    r2, popt = fit_parabola_to_cup(cup_candidate)
    x_data = np.arange(len(cup_candidate))
    fitted_parabola = parabola(x_data, *popt)

    # Plotting with Plotly for smooth rendering and better control
    plotly_fig = go.Figure()

    # Plot background context data (lighter colors)
    plotly_fig.add_trace(go.Candlestick(
        x=context_data['open_time'],
        open=context_data['open'],
        high=context_data['high'],
        low=context_data['low'],
        close=context_data['close'],
        name='Context Data',
        increasing_line_color='darkgreen',
        decreasing_line_color='darkred',
        opacity=0.3
    ))

    # Plot the cup with candlestick chart (highlighted)
    plotly_fig.add_trace(go.Candlestick(
        x=cup_candidate['open_time'],
        open=cup_candidate['open'],
        high=cup_candidate['high'],
        low=cup_candidate['low'],
        close=cup_candidate['close'],
        name='Cup (Candlesticks)',
        increasing_line_color='lime',
        decreasing_line_color='red'
    ))
    
    # Plot the fitted parabola overlay
    plotly_fig.add_trace(go.Scatter(x=cup_candidate['open_time'], y=fitted_parabola, mode='lines', name='Fitted Parabola (Cup)', line=dict(color='cyan', width=3, dash='dash')))

    # Plot the handle with candlesticks (highlighted)
    plotly_fig.add_trace(go.Candlestick(
        x=handle_candidate['open_time'],
        open=handle_candidate['open'],
        high=handle_candidate['high'],
        low=handle_candidate['low'],
        close=handle_candidate['close'],
        name='Handle (Candlesticks)',
        increasing_line_color='yellow',
        decreasing_line_color='orange'
    ))

    # Add vertical lines to mark pattern boundaries
    cup_start_time = cup_candidate.iloc[0]['open_time']
    handle_end_time = handle_candidate.iloc[-1]['open_time']
    
    plotly_fig.add_vline(x=cup_start_time, line=dict(color='blue', width=2, dash='solid'), annotation_text="Cup Start")
    plotly_fig.add_vline(x=handle_end_time, line=dict(color='purple', width=2, dash='solid'), annotation_text="Handle End")

    # If a breakout occurred, highlight the breakout zone
    if breakout_time:
        plotly_fig.add_vline(x=breakout_time, line=dict(color='red', width=3, dash='dot'), annotation_text="Breakout")

    # Convert timestamps for title
    start_time_readable = pd.to_datetime(cup_start_time, unit='ms').strftime('%Y-%m-%d %H:%M')
    end_time_readable = pd.to_datetime(handle_end_time, unit='ms').strftime('%Y-%m-%d %H:%M')

    # Calculate dynamic width based on pattern length
    total_candles = len(cup_candidate) + len(handle_candidate)
    context_candles = min(100, context_after + context_before)  # Context candles shown
    total_displayed_candles = total_candles + context_candles
    
    # Base width calculation: minimum 20px per candle, with reasonable bounds
    min_width = 1200
    max_width = 2400
    calculated_width = max(min_width, min(max_width, total_displayed_candles * 20))
    
    # Set up layout for better visualization
    plotly_fig.update_layout(
        title=f"Pattern {pattern_id}: Cup and Handle ({len(cup_candidate)}+{len(handle_candidate)} candles)<br>Time: {start_time_readable} to {end_time_readable}",
        xaxis_title="Time",
        yaxis_title="Price",
        showlegend=True,
        template="plotly_dark",
        width=calculated_width,
        height=700,
        xaxis=dict(
            type='date',
            tickformat='%H:%M',
            dtick=60000,  # 1-minute intervals for better visibility
            rangeslider=dict(visible=False),  # Hide range slider for cleaner look
            fixedrange=False  # Allow zooming
        )
    )
    
    # Add spacing between candlesticks for better visibility
    plotly_fig.update_traces(
        selector=dict(type='candlestick'),
        line=dict(width=1),  # Thinner candlestick lines
    )
    
    # Adjust the layout to add more spacing (adaptive based on pattern length)
    # Shorter patterns get more spacing, longer patterns get less to avoid over-stretching
    if total_candles <= 40:
        bargap_value = 0.3  # More spacing for short patterns
    elif total_candles <= 80:
        bargap_value = 0.2  # Medium spacing for medium patterns
    else:
        bargap_value = 0.1  # Less spacing for long patterns to keep them readable
        
    plotly_fig.update_layout(
        bargap=bargap_value,  # Adaptive gap between candlesticks
        bargroupgap=0.05  # Small gap between groups
    )

    # Save the plot as a PNG using Kaleido
    # Create patterns directory if it doesn't exist
    os.makedirs("patterns", exist_ok=True)
    
    image_filename = f"cup_handle_{pattern_id}.png"
    image_path = f"patterns/{image_filename}"
    plotly_fig.write_image(image_path, scale=2)  # Scaling the image for better resolution
    print(f"Pattern {pattern_id} plot saved as {image_path}")

def plot_patterns(patterns, data):
    """
    Plot all detected patterns and save as PNG files
    
    Args:
        patterns: List of detected patterns
        data: Original dataset
    """
    # Plot each detected pattern
    for idx, pattern in enumerate(patterns):
        cup_start_idx = pattern['cup_start_idx']
        cup_end_idx = pattern['cup_end_idx']
        handle_start_idx = pattern['handle_start_idx'] 
        handle_end_idx = pattern['handle_end_idx']
        breakout_time = pattern.get('breakout_time', None)

        # Extract cup and handle data based on indices
        cup_candidate = data.iloc[cup_start_idx:cup_end_idx]
        handle_candidate = data.iloc[handle_start_idx:handle_end_idx]
        
        # Plot and save each pattern with the given pattern_id
        plot_and_crop_pattern(
            cup_candidate, 
            handle_candidate, 
            breakout_time, 
            idx + 1, 
            data,  # Pass full dataset for context
            cup_start_idx,  # Pattern start index
            handle_end_idx  # Pattern end index
        )