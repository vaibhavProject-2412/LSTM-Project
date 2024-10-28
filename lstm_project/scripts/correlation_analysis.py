import pandas as pd
from scipy.stats import spearmanr

# Load processed data
def load_data(file_path):
    """Loads the processed data from the CSV file."""
    data = pd.read_csv(file_path)
    return data

# Calculate Spearman correlation
def calculate_spearman_correlation(data, target_column='consumption_scaled'):
    """
    Calculates the Spearman correlation between the target column (e.g., 'consumption_scaled')
    and all other numeric columns in the dataset.
    """
    # Select only numeric columns to calculate correlation
    numeric_columns = data.select_dtypes(include='number').columns.tolist()
    
    # Exclude the target column from the list of features
    features = [col for col in numeric_columns if col != target_column]
    
    correlations = {}
    
    # Calculate Spearman correlation for each feature against the target column
    for feature in features:
        corr, _ = spearmanr(data[feature], data[target_column])
        correlations[feature] = corr
    
    return correlations

# Filter strong correlations
def filter_strong_correlations(correlations, threshold=0.6):
    """Filters features with Spearman correlation above a specified threshold (e.g., |0.6|)."""
    strong_factors = {feature: corr for feature, corr in correlations.items() if abs(corr) >= threshold}
    return strong_factors

# Save correlation results
def save_correlation_results(correlations, output_path):
    """Saves the correlation results to a CSV file."""
    df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Spearman_Correlation'])
    df.to_csv(output_path, index=False)
    print(f"Correlation results saved to {output_path}")

# Main function to perform correlation analysis
def main():
    # Path to the processed data
    data_path = 'data/processed_data.csv'
    
    # Path to save correlation results
    output_path = 'results/spearman_correlation_results.csv'
    
    # Step 1: Load the processed data
    print("Loading processed data...")
    data = load_data(data_path)
    
    # Step 2: Calculate Spearman correlation between features and consumption_scaled
    print("Calculating Spearman correlation...")
    correlations = calculate_spearman_correlation(data)
    
    # Step 3: Filter strong correlations (e.g., |correlation| >= 0.6)
    print("Filtering strong correlations...")
    strong_correlations = filter_strong_correlations(correlations)
    
    # Display strong correlations
    print("Strong correlations:")
    for feature, corr in strong_correlations.items():
        print(f"{feature}: {corr}")
    
    # Step 4: Save correlation results
    save_correlation_results(correlations, output_path)

# Entry point for the script
if __name__ == "__main__":
    main()
