#!/usr/bin/env python3
"""
Unified Cup and Handle Pattern Detection System

This script integrates pattern detection, visualization, and validation
into a single workflow.
"""

import os
import glob
import pandas as pd
from summary_utils import save_validation_reports
from pattern_detector import load_and_prepare_data, detect_cup_and_handle_sequential

MAX_PATTERNS = 30

def load_all_csv_files(data_folder='data'):
    """
    Dynamically search for all CSV files in the data folder and concatenate them
    
    Args:
        data_folder: Path to the folder containing CSV files
        
    Returns:
        pd.DataFrame: Concatenated data from all CSV files, sorted by timestamp
    """
    
    # Find all CSV files in the data folder
    csv_pattern = os.path.join(data_folder, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_folder}' folder")
    
    # Sort files to ensure chronological order
    csv_files.sort()
    
    print(f"📁 Found {len(csv_files)} CSV files:")
    dataframes = []
    total_rows = 0
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        try:
            df = pd.read_csv(csv_file)
            rows = len(df)
            total_rows += rows
            dataframes.append(df)
            print(f"   • {filename}: {rows:,} rows")
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
            continue
    
    if not dataframes:
        raise Exception("No valid CSV files could be loaded")
    
    # Concatenate all dataframes
    print(f"🔄 Concatenating {len(dataframes)} files...")
    combined_data = pd.concat(dataframes, ignore_index=True)
    
    # Sort by timestamp if available
    if 'open_time' in combined_data.columns:
        combined_data = combined_data.sort_values('open_time').reset_index(drop=True)
        print(f"🔗 Data sorted by timestamp")
    
    # Remove any duplicate rows based on timestamp
    if 'open_time' in combined_data.columns:
        initial_rows = len(combined_data)
        combined_data = combined_data.drop_duplicates(subset=['open_time']).reset_index(drop=True)
        duplicates_removed = initial_rows - len(combined_data)
        if duplicates_removed > 0:
            print(f"🧹 Removed {duplicates_removed:,} duplicate rows")
    
    print(f"✅ Combined dataset: {len(combined_data):,} total rows")
    return combined_data

def main():
    """
    Main function that orchestrates the entire cup and handle pattern detection workflow
    """
    print("🔍 Starting Cup and Handle Pattern Detection System...")
    print("=" * 60)
    
    # Step 1: Load and prepare data from all CSV files
    print("📊 Step 1: Loading and preparing data from all CSV files...")
    try:
        # Load all CSV files from data folder
        raw_data = load_all_csv_files('data')
        
        # Prepare data with technical indicators (modify load_and_prepare_data to accept DataFrame)
        from pattern_detector import prepare_data_with_smoothing
        import talib as ta
        
        # Apply smoothing and calculate technical indicators
        data = prepare_data_with_smoothing(raw_data)
        data['ma20'] = data['close'].rolling(window=20).mean()
        data['ATR14'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data.dropna(inplace=True)
        
        print(f"✅ Data prepared successfully! Final dataset contains {len(data):,} rows")
        
        # Show date range
        if 'open_time' in data.columns:
            start_date = pd.to_datetime(data['open_time'].min(), unit='ms').strftime('%Y-%m-%d %H:%M')
            end_date = pd.to_datetime(data['open_time'].max(), unit='ms').strftime('%Y-%m-%d %H:%M')
            print(f"📅 Date range: {start_date} → {end_date}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Detect patterns And Plotting
    print("\n🔍 Step 2: Detecting cup and handle patterns...")
    try:
        patterns = detect_cup_and_handle_sequential(data, max_patterns=MAX_PATTERNS)
        print(f"✅ Pattern detection completed! Found {len(patterns)} patterns")
    except Exception as e:
        print(f"❌ Error during pattern detection: {e}")
        return
    
    if not patterns:
        print("⚠️  No patterns found in the dataset.")
        return
    
    # Step 3: Validate patterns and create reports
    print("\n📋 Step 4: Creating Summary reports...")
    try:        
        # Save reports
        save_validation_reports(
            patterns,
            csv_filename='reports/pattern_summary_report.csv',
            json_filename='reports/pattern_summary_report.json'
        )

        print("✅ Validation reports created successfully!")
            
        print("\n📁 Output Files:")
        print("   • reports/pattern_summary_report.csv - Detailed validation report")
        print("   • reports/pattern_summary_report.json - JSON format report")
        print("   • patterns/cup_handle_*.png - Individual pattern charts")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return
    
    print("\n🎉 Cup and Handle Pattern Detection completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()