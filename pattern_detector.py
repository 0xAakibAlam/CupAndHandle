#!/usr/bin/env python3
"""
Pattern Detector - Main entry point for complete cup and handle pattern detection
Uses modular components for clean separation of concerns
"""

import sys
import time
import threading
import numpy as np
import talib as ta
import pandas as pd
from scipy.signal import savgol_filter
from plot_utils import plot_cup_and_handle_pattern
from scipy.optimize import curve_fit, OptimizeWarning


class ProgressSpinner:
    """
    A rotating progress spinner for terminal output
    """
    def __init__(self, message="Processing"):
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.spinning = False
        self.thread = None
        
    def _spin(self):
        """Internal method to handle the spinning animation"""
        i = 0
        while self.spinning:
            sys.stdout.write(f'\r{self.spinner_chars[i]} {self.message}')
            sys.stdout.flush()
            i = (i + 1) % len(self.spinner_chars)
            time.sleep(0.1)
    
    def start(self, message=None):
        """Start the spinner animation"""
        if message:
            self.message = message
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self, final_message=None):
        """Stop the spinner animation"""
        self.spinning = False
        if self.thread:
            self.thread.join(timeout=0.2)
        
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        
        if final_message:
            print(final_message)
        sys.stdout.flush()


def load_and_prepare_data(filename='data/BTCUSDT-1m-2025-08-14.csv'):
    """
    Load and prepare historical price data with technical indicators
    
    This function loads OHLCV data from a CSV file and calculates several technical indicators:
    - Smoothed close prices using Savitzky-Golay filter (window=15, order=3)
    - 20-period simple moving average
    - 14-period Average True Range for volatility analysis
    - 20-period volume moving average
    
    Args:
        filename: Path to CSV file containing OHLCV data (default: BTCUSDT 1-min data)
        
    Returns:
        DataFrame with original OHLCV data plus calculated technical indicators
    """
    
    # Load raw OHLCV data from CSV
    data = pd.read_csv(filename)
         
    # Apply Savitzky-Golay smoothing to reduce noise while preserving pattern shape
    data['smooth_close'] = savgol_filter(data['close'], 15, 3)

    # Calculate 20-period moving average for trend analysis
    data['ma20'] = data['close'].rolling(window=20).mean()
 
    # Calculate ATR-14 for measuring volatility and potential breakout levels
    data['ATR14'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    
    # Calculate 10-period volume moving average for volume trend analysis
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    
    # Remove rows with NaN values from indicator calculations
    data.dropna(inplace=True)
    
    return data

def parabola(x, a, b, c):
    """Define a parabolic curve (quadratic fit)"""
    return a * x**2 + b * x + c

def prepare_data_with_smoothing(data):
    """
    Apply combined Heikin-Ashi + Savitzky-Golay smoothing for optimal parabola fitting
    
    Args:
        data: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with smooth_close column added
    """
    window_length = 15  # Smaller window for better responsiveness
    polyorder = 3       # Cubic polynomial preserves parabolic shapes
    data['smooth_close'] = savgol_filter(data['close'], window_length, polyorder)
    
    return data

def fit_enhanced_parabola(cup_data):
    """
    Enhanced parabola fitting with multiple improvements for cup detection
    
    Args:
        cup_data: DataFrame containing the cup candidate data
        
    Returns:
        tuple: (r2_score, parabola_parameters)
    """
    
    # Normalize x-data to [0, 1] for numerical stability
    x_data = np.arange(len(cup_data)) / (len(cup_data) - 1)
    y_data = cup_data['smooth_close'].values
    
    # Normalize y-data
    y_min, y_max = y_data.min(), y_data.max()
    y_range = y_max - y_min
    if y_range == 0:  # Flat data
        return -1, [0, 0, 0]
    
    y_normalized = (y_data - y_min) / y_range
    
    best_r2 = -1
    best_params = [0, 0, 0]
    
    # Multiple initial guesses for robust fitting
    initial_guesses = [
        [1.0, -1.0, 0.5],    # Standard U-shape
        [2.0, -2.0, 1.0],    # Steeper U-shape
        [0.5, -0.5, 0.25],   # Shallow U-shape
        [1.5, -1.5, 0.75],   # Medium U-shape
    ]
    
    for p0 in initial_guesses:
        try:
            # Bounds to ensure upward-facing parabola
            bounds = ([0.1, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            
            # Weighted fitting - emphasize rim points and bottom
            weights = np.ones(len(x_data))
            rim_size = max(2, len(x_data) // 10)
            weights[:rim_size] *= 2.0      # Left rim
            weights[-rim_size:] *= 2.0     # Right rim
            
            # Emphasize bottom region
            bottom_idx = np.argmin(y_normalized)
            bottom_range = max(3, len(x_data) // 8)
            bottom_start = max(0, bottom_idx - bottom_range // 2)
            bottom_end = min(len(x_data), bottom_idx + bottom_range // 2)
            weights[bottom_start:bottom_end] *= 1.5
            
            # Curve fitting
            popt, _ = curve_fit(
                parabola, x_data, y_normalized, 
                p0=p0, bounds=bounds, sigma=1/weights, 
                maxfev=2000, absolute_sigma=False
            )
            
            a, b, c = popt
            
            # Validate parabola shape
            if a <= 0:  # Must be upward-facing
                continue
                
            # Calculate fitted values and R²
            fitted = parabola(x_data, *popt)
            ss_res = np.sum(weights * (y_normalized - fitted)**2)
            ss_tot = np.sum(weights * (y_normalized - np.average(y_normalized, weights=weights))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -1

            if r2 > best_r2:
                best_r2 = r2
                best_params = popt.copy()
                
        except (RuntimeError, ValueError, OptimizeWarning):
            continue
        
    return best_r2 if best_r2 > 0 else -1, best_params

def validate_rim_levels(cup_data, tolerance_pct=10.0):
    """
    Validate that left and right rims are at similar price levels
    
    Args:
        cup_data: DataFrame containing the cup candidate data
        tolerance_pct: Percentage tolerance for rim level difference
        
    Returns:
        bool: True if rim levels are similar
    """
    cup_length = len(cup_data)
    left_rim_size = max(3, int(cup_length * 0.1))
    right_rim_size = max(3, int(cup_length * 0.1))
    
    left_rim = cup_data.head(left_rim_size)
    right_rim = cup_data.tail(right_rim_size)
    
    left_rim_high = left_rim['smooth_close'].max()
    right_rim_high = right_rim['smooth_close'].max()
    
    cup_depth = cup_data['smooth_close'].max() - cup_data['smooth_close'].min()
    rim_diff_pct = abs(left_rim_high - right_rim_high) / cup_depth * 100
    
    return rim_diff_pct <= tolerance_pct

def detect_cup_patterns(start_idx, data, min_r2=0.85, min_cup_length=30, max_cup_length=300):
    """
    Detect cup patterns using enhanced parabola fitting
    
    Args:
        data: DataFrame with OHLCV data and smooth_close column
        min_r2: Minimum R² threshold for parabola fit quality
        min_cup_length: Minimum cup length in candles
        max_cup_length: Maximum cup length in candles
        
    Returns:
        list: List of detected cup patterns with metadata
    """
    
    for cup_len in range(min_cup_length, max_cup_length + 1):
        end_idx = start_idx + cup_len
        if end_idx >= len(data):
            break
        
        # Extract cup candidate
        cup_candidate = data.iloc[start_idx:end_idx]
        
        # Fit enhanced parabola
        r2, popt = fit_enhanced_parabola(cup_candidate)
        if r2 < min_r2:
            continue
        
        # Check cup depth (must be meaningful)
        avg_candle_size = (cup_candidate['open'] - cup_candidate['close']).abs().mean()
        cup_depth = cup_candidate['smooth_close'].max() - cup_candidate['smooth_close'].min()
        if cup_depth < 2 * avg_candle_size:
            continue
        
        # Validate rim levels
        if not validate_rim_levels(cup_candidate, tolerance_pct=10.0):
            continue
        
        # Store valid cup pattern
        cup_pattern = {
            'start_time': cup_candidate.iloc[0]['open_time'],
            'end_time': cup_candidate.iloc[-1]['open_time'],
            'cup_depth': cup_depth,
            'r2': r2,
            'parabola_params': popt,
            'cup_start_idx': start_idx,
            'cup_end_idx': end_idx,
            'cup_length': cup_len,
        }

        return cup_pattern
        
    return None

def detect_handle_from_cup_end(data, cup_pattern, max_handle_length=None, min_handle_length=None):
    """
    Detect handle pattern starting from where a cup pattern ends
    
    Args:
        data: Full dataset
        cup_pattern: Cup pattern dictionary from cup_detector
        max_handle_length: Maximum handle length (default: 30% of cup length)
        min_handle_length: Minimum handle length (default: 5% of cup length)
        
    Returns:
        dict or None: Handle pattern dictionary if valid handle found
    """
    cup_end_idx = cup_pattern['cup_end_idx']
    cup_length = cup_pattern['cup_length']
    cup_depth = cup_pattern['cup_depth']
    
    # Calculate dynamic handle length range based on cup length
    if min_handle_length is None:
        min_handle_length = max(5, int(cup_length * 0.05))  # At least 5% of cup
    if max_handle_length is None:
        max_handle_length = max(min_handle_length, min(50, int(cup_length * 0.2)))   # At most 20% of cup
    
    # Get cup data for rim level calculations
    cup_start_idx = cup_pattern['cup_start_idx']
    cup_data = data.iloc[cup_start_idx:cup_end_idx]
    
    # Try different handle lengths
    for handle_length in range(max_handle_length, min_handle_length - 1, -1):
        handle_start_idx = cup_end_idx
        handle_end_idx = handle_start_idx + handle_length
        
        # Check if we have enough data
        if handle_end_idx > len(data):
            continue
            
        # Extract handle candidate
        handle_data = data.iloc[handle_start_idx:handle_end_idx]
        
        # Validate handle
        if validate_handle_pattern(cup_data, handle_data, cup_depth):
            # Return valid handle pattern
            handle_pattern = {
                'handle_start_idx': handle_start_idx,
                'handle_end_idx': handle_end_idx,
                'handle_length': handle_length,
                'handle_depth': handle_data['smooth_close'].max() - handle_data['smooth_close'].min(),
                'handle_high': handle_data['smooth_close'].max(),
                'handle_low': handle_data['smooth_close'].min(),
                'start_time': handle_data.iloc[0]['open_time'],
                'end_time': handle_data.iloc[-1]['open_time'],
                'cup_pattern': cup_pattern  # Reference to the cup
            }

            return handle_pattern
    
    return None  # No valid handle found

def validate_handle_pattern(cup_data, handle_data, cup_depth):
    """
    Validate handle according to cup and handle pattern requirements
    
    Args:
        cup_data: DataFrame containing the cup data
        handle_data: DataFrame containing the handle candidate data
        cup_depth: Depth of the cup pattern
        
    Returns:
        bool: True if handle is valid
    """
    
    # 1. Handle retracement check (should not retrace more than 40% of cup depth)
    handle_depth = handle_data['ma20'].max() - handle_data['ma20'].min()
    if handle_depth / cup_depth > 0.4:
        return False
    
    # 2. Handle high check: Must be lower than cup rims
    handle_high = handle_data['ma20'].max()
    
    # Get rim levels from cup
    cup_length = len(cup_data)
    left_rim_size = max(3, int(cup_length * 0.1))
    right_rim_size = max(3, int(cup_length * 0.1))
    
    left_rim_high = cup_data.head(left_rim_size)['ma20'].max()
    right_rim_high = cup_data.tail(right_rim_size)['ma20'].max()
    max_rim_level = max(left_rim_high, right_rim_high)
    
    if handle_high > max_rim_level:  # Handle high must be strictly lower than rim
        return False
    
    return True

def detect_breakout_from_handle_end(data, handle_pattern, atr_data, max_breakout_distance=20):
    """
    Detect breakout from handle pattern
    
    Args:
        data: Full dataset
        handle_pattern: Handle pattern dictionary
        atr_data: ATR values for breakout validation
        max_breakout_distance: Maximum distance to look for breakout
        
    Returns:
        dict or None: Breakout information if found
    """
    handle_end_idx = handle_pattern['handle_end_idx']
    handle_high = handle_pattern['handle_high']
    
    # Calculate breakout threshold: handle high + 1.5x ATR
    atr_value = atr_data.iloc[handle_end_idx] if handle_end_idx < len(atr_data) else atr_data.iloc[-1]
    breakout_threshold = handle_high + (1.5 * atr_value)
    
    # Search for breakout within reasonable distance after handle
    search_end_idx = min(len(data), handle_end_idx + max_breakout_distance)
    data_slice = data.iloc[handle_end_idx:search_end_idx]
    
    # Find first candle that closes above the breakout threshold
    breakout_candidates = data_slice[data_slice['ma20'] > breakout_threshold]
    if breakout_candidates.empty:
        return None  # No breakout detected
    
    # Get breakout details
    first_breakout_idx = breakout_candidates.index[0]
    breakout_candle = data.loc[first_breakout_idx]
    
    breakout_info = {
        'breakout_time': breakout_candle['open_time'],
        'breakout_price': breakout_candle['ma20'],
        'breakout_threshold': breakout_threshold,
        'breakout_strength': breakout_candle['ma20'] - handle_high,
        'breakout_strength_pct': (breakout_candle['ma20'] - handle_high) / handle_high * 100,
        'atr_multiplier': (breakout_candle['ma20'] - handle_high) / atr_value,
        'breakout_idx': first_breakout_idx
    }
    
    return breakout_info

def detect_cup_and_handle_sequential(data, max_patterns=10, min_r2=0.85, plot_immediately=True):
    """
    Sequential left-to-right detection: Find cup → Detect handle → Plot immediately
    
    Args:
        data: DataFrame with OHLCV data and smooth_close column
        max_patterns: Maximum number of patterns to detect
        min_r2: Minimum R² threshold for cup quality
        plot_immediately: Whether to plot each pattern immediately when found
        
    Returns:
        list: List of complete cup and handle patterns
    """    
    complete_patterns = []
    start_idx = 0
    min_cup_length = 30
    max_cup_length = 300
    avg_candle_size = (data['high'] - data['low']).mean()
    
    # Initialize progress spinner
    spinner = ProgressSpinner()
    total_scans = 0
    candlesticks_processed = 0
    
    print("🔍 Sequential Detection: Scanning left to right...")
    
    # Start the spinner for the first pattern search
    spinner.start(f"🔄 Searching for pattern #{len(complete_patterns)+1} | Processed: {candlesticks_processed:,} candles")
    
    while start_idx < len(data) - min_cup_length - 50 and len(complete_patterns) < max_patterns:
        total_scans += 1
        candlesticks_processed = start_idx
        
        # Update spinner message frequently for real-time feedback
        if total_scans % 10 == 0:  # Update every 10 iterations instead of 100
            progress_pct = (start_idx / (len(data) - min_cup_length - 50)) * 100
            
            # Only parse timestamp every 100 iterations to avoid performance hit
            if total_scans % 100 == 0:
                current_time = pd.to_datetime(data.iloc[start_idx]['open_time'], unit='ms').strftime('%H:%M:%S')
                spinner.stop()
                spinner.start(f"🔄 Searching for pattern #{len(complete_patterns)+1} | Processed: {candlesticks_processed:,}/{len(data):,} candles ({progress_pct:.1f}%)")
            else:
                # Quick update without timestamp parsing
                spinner.stop()
                spinner.start(f"🔄 Searching for pattern #{len(complete_patterns)+1} | Processed: {candlesticks_processed:,}/{len(data):,} candles ({progress_pct:.1f}%)")
        
        # Step 1: Try to find a cup at current position
        cup_pattern = detect_cup_patterns(start_idx, data, min_r2=min_r2, min_cup_length=min_cup_length, max_cup_length=max_cup_length)
        if not cup_pattern:
            start_idx += 10
            continue
        
        # Step 2: Try to detect handle from cup end
        handle_pattern = detect_handle_from_cup_end(data, cup_pattern)
        if not handle_pattern:
            start_idx += 10
            continue
            
        # Step 3: Try to find breakout from handle
        breakout_info = detect_breakout_from_handle_end(data, handle_pattern, data['ATR14'])
        if breakout_info is None:
            start_idx += 10
            continue
        
        # Stop spinner - we found a complete pattern!
        spinner.stop()
                
        # Create complete pattern
        complete_pattern = {
            'pattern_id': len(complete_patterns) + 1,
            'cup': cup_pattern,
            'handle': handle_pattern,
            'breakout': breakout_info,
            'total_length': cup_pattern['cup_length'] + handle_pattern['handle_length'],
            'start_time': cup_pattern['start_time'],
            'end_time': breakout_info['breakout_time'],
            'cup_start_idx': cup_pattern['cup_start_idx'],
            'cup_end_idx': cup_pattern['cup_end_idx'],
            'handle_start_idx': handle_pattern['handle_start_idx'],
            'handle_end_idx': handle_pattern['handle_end_idx'],
            'breakout_idx': breakout_info['breakout_idx']
        }
        
        complete_patterns.append(complete_pattern)

        # Show pattern properties
        pattern_start_time = pd.to_datetime(cup_pattern['start_time'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        pattern_end_time = pd.to_datetime(breakout_info['breakout_time'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"🎉 COMPLETE PATTERN #{len(complete_patterns)} FOUND!")
        print(f"   📍 Period: {pattern_start_time} → {pattern_end_time}")
        print(f"   🏺 Cup: R²={cup_pattern['r2']:.3f}, Length={cup_pattern['cup_length']} candles, Depth={cup_pattern['cup_depth']:.2f}")
        print(f"   🔧 Handle: Length={handle_pattern['handle_length']} candles, Depth={handle_pattern['handle_depth']:.2f}")
        print(f"   🚀 Breakout: Strength={breakout_info['breakout_strength_pct']:.2f}%, Price={breakout_info['breakout_price']:.2f}")

        # Step 4: Plot immediately if requested
        if plot_immediately:
            print(f"   📊 Generating chart...")
            plot_cup_and_handle_pattern(data, complete_pattern)
        
        print(f"   ✅ Pattern #{len(complete_patterns)} processed successfully!")
        print()
        
        start_idx = handle_pattern['handle_end_idx']
        
        # Start spinner for next pattern search (if we haven't reached the limit)
        if len(complete_patterns) < max_patterns and start_idx < len(data) - min_cup_length - 50:
            candlesticks_processed = start_idx
            progress_pct = (start_idx / (len(data) - min_cup_length - 50)) * 100
            spinner.start(f"🔄 Searching for pattern #{len(complete_patterns)+1} | Processed: {candlesticks_processed:,}/{len(data):,} candles ({progress_pct:.1f}%)")
    
    # Final cleanup
    spinner.stop()
    
    if len(complete_patterns) == 0:
        print("❌ No patterns found in the dataset")
    else:
        print(f"✅ Detection completed! Found {len(complete_patterns)} patterns")
    
    return complete_patterns