import pandas as pd
import numpy as np

# Load the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}. Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("First 5 rows:\n", df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Handle missing values
def handle_missing_values(df):
    print("\nMissing values before handling:\n", df.isnull().sum())
    # Fill missing 'director' and 'cast' with 'Unknown'
    df["director"].fillna("Unknown", inplace=True)
    df["cast"].fillna("Unknown", inplace=True)
    # Fill missing 'country' with the mode
    df["country"].fillna(df["country"].mode()[0], inplace=True)
    # Drop rows with missing 'rating' or 'date_added' as they are critical
    df.dropna(subset=["rating", "date_added"], inplace=True)
    print("\nMissing values after handling:\n", df.isnull().sum())
    return df

# Convert 'date_added' to datetime and extract 'month_added' and 'year_added'
def process_date_added(df):
    # Strip whitespace before converting to datetime
    df["date_added"] = df["date_added"].str.strip()
    df["date_added"] = pd.to_datetime(df["date_added"], format="%B %d, %Y")
    df["month_added"] = df["date_added"].dt.month
    df["year_added"] = df["date_added"].dt.year
    return df

# Clean 'duration' column
def clean_duration(df):
    df["duration"] = df["duration"].apply(lambda x: int(x.split(" ")[0]) if "min" in x else int(x.split(" ")[0])*50)
    return df

# Function to save processed data
def save_data(df, output_path, description="processed data"):
    try:
        df.to_csv(output_path, index=False)
        print(f"\n{description.capitalize()} saved successfully to {output_path}")
        print(f"Saved dataset shape: {df.shape}")
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")

# Main execution
if __name__ == "__main__":
    # Update the file path to match your actual file
    file_path = "netflix_titles.csv"  # Changed from "../data/NETFLIXMOVIESANDTVSHOWSCLUSTERING.csv"
    
    netflix_df = load_data(file_path)
    
    if netflix_df is not None:
        netflix_df = handle_missing_values(netflix_df)
        netflix_df = process_date_added(netflix_df)
        netflix_df = clean_duration(netflix_df)
        
        print("\nDataFrame after preprocessing:\n", netflix_df.head())
        print("\nDataFrame Info:\n")
        netflix_df.info()

        # Save the preprocessed data
        output_path = "netflix_preprocessed.csv"  # Changed from "../data/netflix_preprocessed.csv"
        save_data(netflix_df, output_path, "preprocessed data")
        
        # Optional: Save additional versions of the data
        # save_data(netflix_df, "netflix_cleaned.csv", "cleaned data")
        # save_data(netflix_df[['title', 'type', 'rating', 'duration']], "netflix_summary.csv", "summary data")
    else:
        print("Failed to load data. Please check the file path and try again.")

