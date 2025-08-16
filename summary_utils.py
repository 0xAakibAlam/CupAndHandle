import pandas as pd
import os

def validate_patterns(patterns, data):
    """
    Validate detected patterns and create a summary report
    
    Args:
        patterns: List of detected patterns
        data: Original dataset
    
    Returns:
        DataFrame with validation summary
    """
    # Create a list to store the validation summary for each pattern
    pattern_summary = []
    
    # Calculate average candle size for validation
    avg_candle_size = (data['high'] - data['low']).mean()
    
    # Loop through the detected patterns to extract required details
    for pattern in patterns:
        # Extract pattern details
        pattern_id = patterns.index(pattern) + 1
        cup_start_time = pattern['start_time']
        handle_end_time = pattern['end_time']
        cup_depth = pattern['cup_depth']
        handle_depth = pattern['handle_depth']
        r2_value = pattern['r2']
        breakout_time = pattern.get('breakout_time', None)
        
        # Validate the pattern
        if r2_value < 0.85:
            validity = "Invalid"
            reason = "Low R² value (not smooth enough)"
        elif cup_depth < 2 * avg_candle_size:
            validity = "Invalid"
            reason = "Cup depth is too shallow"
        elif handle_depth / cup_depth > 0.4:
            validity = "Invalid"
            reason = "Handle retraced more than 40% of cup depth"
        elif breakout_time is None:
            validity = "Invalid"
            reason = "No breakout detected"
        else:
            validity = "Valid"
            reason = "Pattern is valid"
        
        # Store pattern information in the summary list
        pattern_summary.append({
            'pattern_id': pattern_id,
            'start_time': cup_start_time,
            'end_time': handle_end_time,
            'cup_depth': cup_depth,
            'handle_depth': handle_depth,
            'r2_value': r2_value,
            'breakout_time': breakout_time,
            'validity': validity,
            'reason': reason
        })
    
    return pd.DataFrame(pattern_summary)

def save_validation_reports(df_summary, csv_filename='reports/pattern_summary_report.csv', json_filename='reports/pattern_summary_report.json'):
    """
    Save validation reports as CSV and JSON files
    
    Args:
        df_summary: DataFrame with validation summary
        csv_filename: Name of CSV file to save
        json_filename: Name of JSON file to save
    """
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Save the summary as a CSV file
    df_summary.to_csv(csv_filename, index=False)
    print(f"Pattern summary report saved as '{csv_filename}'")
    
    # Save as JSON (proper JSON array format)
    df_summary.to_json(json_filename, orient='records', indent=2)
    print(f"Pattern summary report saved as '{json_filename}'")
