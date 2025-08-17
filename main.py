#!/usr/bin/env python3
"""
Unified Cup and Handle Pattern Detection System

This script integrates pattern detection, visualization, and validation
into a single workflow.
"""

from summary_utils import save_validation_reports
from pattern_detector import load_and_prepare_data, detect_cup_and_handle_sequential

MAX_PATTERNS = 30
file_name = 'data/BTCUSDT-1m-2025-07.csv'

def main():
    """
    Main function that orchestrates the entire cup and handle pattern detection workflow
    """
    print("🔍 Starting Cup and Handle Pattern Detection System...")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("📊 Step 1: Loading and preparing data...")
    try:
        data = load_and_prepare_data(file_name)
        print(f"✅ Data loaded successfully! Dataset contains {len(data)} rows")
    except FileNotFoundError:
        print("❌ Error: CSV file not found. Please check the filename.")
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