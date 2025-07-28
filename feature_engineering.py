import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Load the preprocessed dataset
def load_preprocessed_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Feature Engineering
def feature_engineer(df):
    # Clean 'duration' column first - convert to numeric
    def clean_duration_value(x):
        if pd.isna(x):
            return 0
        if "Season" in str(x):
            # Convert seasons to minutes (assuming 10 hours per season = 600 minutes)
            return int(str(x).split(" ")[0]) * 600
        elif "min" in str(x):
            # Extract minutes
            return int(str(x).split(" ")[0])
        else:
            # Default fallback
            return 90  # Average movie length
    
    df["duration"] = df["duration"].apply(clean_duration_value)
    
    # Content age
    df["content_age"] = 2025 - df["release_year"]

    # Genre count
    df["genre_count"] = df["listed_in"].apply(lambda x: len(x.split(", ")))

    # TF-IDF for description
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df["description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.index = df.index
    df = pd.concat([df, tfidf_df], axis=1)

    # One-hot encode 'type', 'rating', 'country'
    df = pd.get_dummies(df, columns=["type"], prefix="type")
    df = pd.get_dummies(df, columns=["rating"], prefix="rating")

    # Multi-label binarization for 'listed_in' (genres)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df["listed_in"].apply(lambda x: x.split(", ")))
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    genres_df.index = df.index
    df = pd.concat([df, genres_df], axis=1)

    # Drop original categorical columns that have been encoded or are not needed for clustering
    df_encoded = df.drop(columns=["show_id", "title", "director", "cast", "country", "date_added", "listed_in", "description", "release_year", "month_added", "year_added"], errors="ignore")

    print("\nDataFrame after feature engineering:\n", df_encoded.head())
    print("\nDataFrame Info after feature engineering:\n")
    df_encoded.info()
    return df_encoded

# Main execution
if __name__ == "__main__":
    # Specify the CSV file you want to load
    file_path = "netflix_preprocessed.csv"  # Change this to your desired CSV file
    
    netflix_df = load_preprocessed_data(file_path)
    
    if netflix_df is not None:
        netflix_features = feature_engineer(netflix_df)
        
        # Save the featured data for later use
        netflix_features.to_csv("netflix_features.csv", index=False)
        print("\nFeatured data saved to netflix_features.csv")
    else:
        print("Failed to load data. Please check the file path and try again.")