import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load datasets
def load_data(smart_meter_path, household_info_path):
    """Loads smart meter and household information data from CSV files."""
    smart_meter_data = pd.read_csv(smart_meter_path)
    household_info_data = pd.read_csv(household_info_path)
    return smart_meter_data, household_info_data

# Merge datasets
def merge_data(smart_meter_data, household_info_data):
    """Merges the smart meter data with household information data on LCLid."""
    merged_data = pd.merge(smart_meter_data, household_info_data, on='LCLid')
    return merged_data

# Scale the electricity consumption data
def scale_data(merged_data, consumption_column='energy_sum'):
    """Scales the consumption column for use in the forecasting model."""
    print(f"Scaling '{consumption_column}' column.")
    
    if consumption_column not in merged_data.columns:
        raise ValueError(f"'{consumption_column}' column not found in the merged data.")
    
    # Check if the column is empty
    if merged_data[consumption_column].isnull().all():
        raise ValueError(f"The column '{consumption_column}' is empty or contains only NaN values.")
    
    scaler = StandardScaler()
    merged_data['consumption_scaled'] = scaler.fit_transform(merged_data[[consumption_column]])
    return merged_data, scaler

# Save processed data
def save_data(merged_data, output_path):
    """Saves the merged and preprocessed data to a CSV file."""
    merged_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Main function to execute the preprocessing steps
def main():
    # Paths to the manually downloaded datasets
    smart_meter_path = 'data/daily_dataset.csv'
    household_info_path = 'data/informations_households.csv'
    
    # Path to save the processed data
    output_path = 'data/processed_data.csv'
    
    # Step 1: Load datasets
    print("Loading datasets...")
    smart_meter_data, household_info_data = load_data(smart_meter_path, household_info_path)
    
    # Step 2: Merge datasets on LCLid
    print("Merging datasets on 'LCLid'...")
    merged_data = merge_data(smart_meter_data, household_info_data)
    
    # Step 3: Scale consumption data (using energy_sum as the consumption column)
    print("Scaling consumption data...")
    merged_data, scaler = scale_data(merged_data)
    
    # Step 4: Save processed data
    save_data(merged_data, output_path)

# Entry point for the script
if __name__ == "__main__":
    main()
