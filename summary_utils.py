import pandas as pd
import os
from datetime import datetime

def create_pattern_summary(patterns):
    """
    Create a comprehensive summary DataFrame from detected patterns
    
    Args:
        patterns: List of detected pattern dictionaries
        
    Returns:
        pd.DataFrame: Summary DataFrame with all required statistics
    """
    summary_data = []
    
    for pattern in patterns:
        try:
            # Extract pattern components
            pattern_id = pattern.get('pattern_id', 'Unknown')
            cup_pattern = pattern.get('cup', {})
            handle_pattern = pattern.get('handle', {})
            breakout_info = pattern.get('breakout', None)
            
            # Basic pattern info
            start_time = cup_pattern.get('start_time', None)
            end_time = handle_pattern.get('end_time', None)
            
            # Cup statistics
            cup_depth = cup_pattern.get('cup_depth', 0)
            cup_duration = cup_pattern.get('cup_length', 0)
            r2_value = cup_pattern.get('r2', 0)
            
            # Handle statistics
            handle_depth = handle_pattern.get('handle_depth', 0)
            handle_duration = handle_pattern.get('handle_length', 0)
            
            # Breakout information
            breakout_timestamp = None
            breakout_strength = None
            if breakout_info:
                breakout_timestamp = breakout_info.get('breakout_time', None)
                breakout_strength = breakout_info.get('breakout_strength_pct', 0)
            
            # Validation status
            valid_flag = 'Valid'
            invalid_reason = 'N/A'
            
            # Basic validation checks
            if r2_value < 0.85:
                valid_flag = 'Invalid'
                invalid_reason = f'Poor parabolic fit (R²={r2_value:.3f} < 0.85)'
            elif cup_depth <= 0:
                valid_flag = 'Invalid'
                invalid_reason = 'Invalid cup depth'
            elif handle_duration < 5:
                valid_flag = 'Invalid'
                invalid_reason = f'Handle too short ({handle_duration} candles < 5)'
            elif not breakout_info:
                valid_flag = 'Incomplete'
                invalid_reason = 'No breakout detected'
            
            # Convert timestamps to readable format
            start_time_str = pd.to_datetime(start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A'
            end_time_str = pd.to_datetime(end_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S') if end_time else 'N/A'
            breakout_time_str = pd.to_datetime(breakout_timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S') if breakout_timestamp else 'N/A'
            
            # Create summary record
            summary_record = {
                'Pattern_ID': pattern_id,
                'Start_Time': start_time_str,
                'End_Time': end_time_str,
                'Cup_Depth': round(cup_depth, 4),
                'Cup_Duration_Candles': cup_duration,
                'Handle_Depth': round(handle_depth, 4),
                'Handle_Duration_Candles': handle_duration,
                'R_Squared': round(r2_value, 4),
                'Breakout_Timestamp': breakout_time_str,
                'Breakout_Strength_Pct': round(breakout_strength, 2) if breakout_strength else 'N/A',
                'Valid_Flag': valid_flag,
                'Invalid_Reason': invalid_reason,
                'Total_Duration_Candles': cup_duration + handle_duration,
                'Cup_Handle_Ratio': round(cup_duration / handle_duration, 2) if handle_duration > 0 else 'N/A'
            }
            
            summary_data.append(summary_record)
            
        except Exception as e:
            # Handle any errors in processing individual patterns
            summary_record = {
                'Pattern_ID': pattern.get('pattern_id', 'Error'),
                'Start_Time': 'Error',
                'End_Time': 'Error',
                'Cup_Depth': 0,
                'Cup_Duration_Candles': 0,
                'Handle_Depth': 0,
                'Handle_Duration_Candles': 0,
                'R_Squared': 0,
                'Breakout_Timestamp': 'Error',
                'Breakout_Strength_Pct': 'Error',
                'Valid_Flag': 'Invalid',
                'Invalid_Reason': f'Processing error: {str(e)}',
                'Total_Duration_Candles': 0,
                'Cup_Handle_Ratio': 'Error'
            }
            summary_data.append(summary_record)
    
    return pd.DataFrame(summary_data)

def save_validation_reports(patterns, csv_filename='reports/pattern_summary_report.csv', json_filename='reports/pattern_summary_report.json'):
    """
    Save validation reports as CSV and JSON files
    
    Args:
        patterns: List of detected pattern dictionaries
        csv_filename: Name of CSV file to save
        json_filename: Name of JSON file to save
    """
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Create comprehensive summary DataFrame
    df_summary = create_pattern_summary(patterns)
    
    # Save the summary as a CSV file
    df_summary.to_csv(csv_filename, index=False)
    
    # Save as JSON (proper JSON array format)
    df_summary.to_json(json_filename, orient='records', indent=2)
    
    # Print summary statistics
    total_patterns = len(df_summary)
    valid_patterns = len(df_summary[df_summary['Valid_Flag'] == 'Valid'])
    invalid_patterns = len(df_summary[df_summary['Valid_Flag'] == 'Invalid'])
    incomplete_patterns = len(df_summary[df_summary['Valid_Flag'] == 'Incomplete'])
    
    print(f"\n📈 Summary Statistics:")
    print(f"   • Total Patterns: {total_patterns}")
    print(f"   • Valid Patterns: {valid_patterns}")
    print(f"   • Invalid Patterns: {invalid_patterns}")
    print(f"   • Incomplete Patterns: {incomplete_patterns}")
    
    if valid_patterns > 0:
        avg_r2 = df_summary[df_summary['Valid_Flag'] == 'Valid']['R_Squared'].mean()
        avg_cup_depth = df_summary[df_summary['Valid_Flag'] == 'Valid']['Cup_Depth'].mean()
        avg_breakout_strength = df_summary[(df_summary['Valid_Flag'] == 'Valid') & (df_summary['Breakout_Strength_Pct'] != 'N/A')]['Breakout_Strength_Pct'].mean()
        
        print(f"   • Average R²: {avg_r2:.3f}")
        print(f"   • Average Cup Depth: {avg_cup_depth:.2f}")
        if not pd.isna(avg_breakout_strength):
            print(f"   • Average Breakout Strength: {avg_breakout_strength:.2f}%")
    
    return df_summary