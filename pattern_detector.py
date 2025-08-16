import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import talib as ta

def load_and_prepare_data(filename='data/BTCUSDT-1m-2025-08-14.csv'):
    """Load and prepare the dataset for pattern detection"""
    # Load your dataset
    data = pd.read_csv(filename)
    
    # Smooth the data using a moving average to reduce noise
    data['smooth_close'] = data['close'].rolling(window=20).mean()
    
    # Calculate ATR (14) for breakout validation
    data['ATR14'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    
    # Drop rows with NaN values (created by moving averages and ATR calculation)
    data.dropna(inplace=True)
    
    return data

# Define a parabolic curve (quadratic fit)
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

# Function to fit a parabola and return the R² value
def fit_parabola_to_cup(cup_data):
    x_data = np.arange(len(cup_data))
    y_data = cup_data['smooth_close'].values
    
    try:
        # Fit parabola with initial guess for upward-facing curve
        popt, _ = curve_fit(parabola, x_data, y_data, p0=[0.1, 1, 1])
        
        # Check if parabola opens upward (a > 0)
        a, b, c = popt
        if a <= 0:
            # If parabola opens downward or is flat, this is not a valid cup
            return -1, popt  # Return negative R² to indicate invalid
        
        fitted = parabola(x_data, *popt)
        r2 = 1 - np.sum((y_data - fitted)**2) / np.sum((y_data - np.mean(y_data))**2)  # Calculate R²
        
        # Additional validation: check if the cup actually has a U-shape
        # The lowest point should be roughly in the middle third of the pattern
        lowest_idx = np.argmin(fitted)
        pattern_length = len(x_data)
        if lowest_idx < pattern_length * 0.2 or lowest_idx > pattern_length * 0.8:
            # If the minimum is too close to the edges, it's not a proper cup
            return -1, popt
            
        return r2, popt
        
    except Exception as e:
        # If curve fitting fails, return invalid result
        return -1, [0, 0, 0]

# Function to detect handle and check its validity
def detect_handle(cup_data, handle_data, cup_depth):
    # Handle retracement check (should not retrace more than 40% of cup depth)
    handle_depth = handle_data['smooth_close'].max() - handle_data['smooth_close'].min()
    if handle_depth / cup_depth > 0.4:
        return False
    
    # Handle high check: Should be below or equal to cup rim (left/right)
    handle_high = handle_data['smooth_close'].max()
    if handle_high > cup_data['smooth_close'].max():
        return False
    
    # Handle slope check: Should be downward or sideways
    handle_slope = np.polyfit(np.arange(len(handle_data)), handle_data['smooth_close'], 1)[0]  # Linear fit slope
    if handle_slope > 0:  # Upward slope is invalid
        return False
    
    return True

# Function to detect breakout
def detect_breakout(handle_data, atr_data, handle_end_idx, data, max_breakout_distance=20):
    handle_high = handle_data['smooth_close'].max()
    breakout_threshold = handle_high + 1.5 * atr_data.iloc[handle_end_idx]  # 1.5 * ATR(14)
    
    # Limit the search window to reasonable distance after handle
    # Breakout should occur within max_breakout_distance candles after handle ends
    search_end_idx = min(len(data), handle_end_idx + max_breakout_distance)
    data_slice = data.iloc[handle_end_idx:search_end_idx]
    
    # Find breakout candidates within the limited timeframe
    breakout_candidates = data_slice[data_slice['close'] > breakout_threshold]
    if breakout_candidates.empty:
        return None  # No breakout detected within reasonable timeframe
    
    # Additional validation: ensure the breakout is sustained
    # Check that the breakout candle is followed by at least one more candle above the threshold
    first_breakout_idx = breakout_candidates.index[0]
    
    # Look for confirmation in the next few candles
    confirmation_window = min(3, len(data) - first_breakout_idx - 1)
    if confirmation_window > 0:
        next_candles = data.iloc[first_breakout_idx + 1:first_breakout_idx + 1 + confirmation_window]
        # At least one of the next candles should also be above the threshold
        if next_candles.empty or not any(next_candles['close'] > breakout_threshold):
            return None  # Breakout not sustained
    
    breakout_time = breakout_candidates.iloc[0]['open_time']
    return breakout_time

def detect_patterns(data, max_patterns=10):
    """Detect cup and handle patterns in the data"""
    min_cup_len = 30
    max_cup_len = 300
    patterns = []
    start_idx = 0

    while start_idx < len(data) - min_cup_len - 50 and len(patterns) < max_patterns:
        pattern_found = False
        
        for cup_len in range(min_cup_len, max_cup_len + 1):
            if len(patterns) >= max_patterns:  # Stop if we have enough patterns
                break
                
            end_idx = start_idx + cup_len
            if end_idx >= len(data):
                break
            
            # Extract cup candidate
            cup_candidate = data.iloc[start_idx:end_idx]
            
            # Fit parabola and check R²
            r2, popt = fit_parabola_to_cup(cup_candidate)
            if r2 < 0.85:  # Cup must have smooth upward-facing parabolic shape (R² > 0.85)
                continue
            
            # Check cup depth (must be at least 2x the average candle size)
            cup_depth = cup_candidate['smooth_close'].max() - cup_candidate['smooth_close'].min()
            avg_candle_size = (data['high'] - data['low']).mean()
            if cup_depth < 2 * avg_candle_size:
                continue
            
            # Look for handle immediately after cup
            handle_start = end_idx
            for handle_len in range(5, 51):  # Check handle lengths from 5 to 50 candles
                if len(patterns) >= max_patterns:  # Stop if we have enough patterns
                    break
                    
                if handle_start + handle_len > len(data):  # Don't exceed data length
                    break

                handle_candidate = data.iloc[handle_start:handle_start + handle_len]
                
                # Validate handle
                if not detect_handle(cup_candidate, handle_candidate, cup_depth):
                    continue  # Invalid handle
                
                # Detect breakout
                breakout_time = detect_breakout(handle_candidate, data['ATR14'], handle_start + handle_len - 1, data)
                if breakout_time is None:
                    continue  # No breakout detected
                
                # If all conditions pass, store the pattern
                pattern = {
                    'start_time': cup_candidate.iloc[0]['open_time'],
                    'end_time': handle_candidate.iloc[-1]['open_time'],
                    'cup_depth': cup_depth,
                    'handle_depth': handle_candidate['smooth_close'].max() - handle_candidate['smooth_close'].min(),
                    'r2': r2,
                    'breakout_time': breakout_time,
                    'cup_start_idx': start_idx,
                    'cup_end_idx': end_idx,
                    'handle_start_idx': handle_start,
                    'handle_end_idx': handle_start + handle_len
                }
                patterns.append(pattern)
                print(f"Valid pattern detected: Cup start at {cup_candidate.iloc[0]['open_time']} with breakout at {breakout_time}")
                print(f"Cup length: {cup_len}")
                print(f"Handle length: {handle_len}")

                # Move start_idx to after this pattern to avoid overlap
                start_idx = handle_start + handle_len
                pattern_found = True
                break
            
            if pattern_found:
                break
        
        # If no pattern found at this position, advance by a small step
        if not pattern_found:
            start_idx += 10  # Advance by 10 candles to find the next potential pattern

    return patterns

# Main execution when script is run directly
if __name__ == "__main__":
    data = load_and_prepare_data()
    patterns = detect_patterns(data)
    
    # Display the patterns
    for pattern in patterns:
        print(pattern)