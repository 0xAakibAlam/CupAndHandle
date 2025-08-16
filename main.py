#!/usr/bin/env python3
"""
Unified Cup and Handle Pattern Detection System

This script integrates pattern detection, visualization, and validation
into a single workflow.
"""

import pandas as pd
from pattern_detector import load_and_prepare_data, detect_patterns
from plot_utils import plot_patterns
from summary_utils import validate_patterns, save_validation_reports

def main():
    """
    Main function that orchestrates the entire cup and handle pattern detection workflow
    """
    print("🔍 Starting Cup and Handle Pattern Detection System...")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("📊 Step 1: Loading and preparing data...")
    try:
        data = load_and_prepare_data('data/BTCUSDT-1m-2025-07.csv')
        print(f"✅ Data loaded successfully! Dataset contains {len(data)} rows")
    except FileNotFoundError:
        print("❌ Error: CSV file not found. Please check the filename.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Detect patterns
    print("\n🔍 Step 2: Detecting cup and handle patterns...")
    try:
        patterns = detect_patterns(data, max_patterns=10)
        print(f"✅ Pattern detection completed! Found {len(patterns)} patterns")
    except Exception as e:
        print(f"❌ Error during pattern detection: {e}")
        return
    
    if not patterns:
        print("⚠️  No patterns found in the dataset.")
        return
    
    # Step 3: Generate visualizations
    print("\n📈 Step 3: Generating pattern visualizations...")
    try:
        plot_patterns(patterns, data)
        print(f"✅ Generated {len(patterns)} pattern charts")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        print("⚠️  Continuing without plots...")
    
    # Step 4: Validate patterns and create reports
    print("\n📋 Step 4: Validating patterns and creating reports...")
    try:
        validation_summary = validate_patterns(patterns, data)
        
        # Save reports
        save_validation_reports(
            validation_summary,
            csv_filename='reports/pattern_summary_report.csv',
            json_filename='reports/pattern_summary_report.json'
        )
        
        print("✅ Validation reports created successfully!")
        
        # Display summary statistics
        valid_patterns = validation_summary[validation_summary['validity'] == 'Valid']
        invalid_patterns = validation_summary[validation_summary['validity'] == 'Invalid']
        
        print(f"\n📊 Summary Statistics:")
        print(f"   • Total patterns detected: {len(patterns)}")
        print(f"   • Valid patterns: {len(valid_patterns)}")
        print(f"   • Invalid patterns: {len(invalid_patterns)}")
        
        if len(valid_patterns) > 0:
            print(f"   • Average R² value: {valid_patterns['r2_value'].mean():.4f}")
            print(f"   • Average cup depth: {valid_patterns['cup_depth'].mean():.2f}")
            
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
