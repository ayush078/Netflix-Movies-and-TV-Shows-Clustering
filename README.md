## 📋 Table of Contents
* [Project Description](#-project-description)
* [Methodology](#-methodology)
* [Dataset](#-dataset)
* [Installation](#-installation)
* [Usage](#-usage)
* [File Descriptions](#-file-descriptions)

---

## 📝 Project Description

This project performs the following key steps:
1.  **Loads** the Netflix dataset.
2.  **Preprocesses and cleans** the text data by handling missing values.
3.  **Engineers a combined feature** by merging text from multiple columns (`title`, `director`, `cast`, `listed_in`, `description`).
4.  **Converts** the combined text into a numerical format using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.
5.  **Applies the K-Means clustering algorithm** to group the data into 5 distinct clusters.
6.  **Reduces the dimensionality** of the TF-IDF matrix using **PCA** to enable 2D visualization.
7.  **Visualizes** the resulting clusters on a scatter plot and saves the output.

## 🛠 Methodology

The core of this project is an unsupervised learning pipeline:

1.  **Feature Engineering**: A new feature, `combined_features`, is created by concatenating all relevant text fields. This provides a rich "document" for each title, capturing information about its plot, creators, and genre in one place.

2.  **TF-IDF Vectorization**: We use `TfidfVectorizer` from `scikit-learn` to transform the text documents into a matrix of TF-IDF features. This method effectively converts text into a meaningful numerical representation by weighting words based on their importance. Common English "stop words" are removed.

3.  **K-Means Clustering**: The `KMeans` algorithm is used to partition the data into a pre-specified number of clusters (in this case, 5). It works by iteratively assigning each data point to the nearest cluster centroid and then recalculating the centroid's position. We use `k-means++` for smarter initialization and set `n_init=10` to ensure a robust result.

4.  **PCA for Visualization**: Since the TF-IDF matrix has thousands of dimensions (one for each word), we cannot visualize it directly. Principal Component Analysis (PCA) is used to reduce these dimensions to just two, preserving as much of the original variance as possible.

## 📊 Dataset

The dataset used is `netflix_titles.csv`, which contains information about movies and TV shows available on Netflix. The key columns utilized in this project are:
* `title`
* `director`
* `cast`
* `listed_in` (genres)
* `description`

## ⚙️ Installation

To set up the environment and run this project, follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

There are two main scripts in this project.

1.  **Exploratory Data Analysis (Optional):**
    To get a basic understanding of the dataset (info, missing values, etc.), run the `eda.py` script.
    ```bash
    python eda.py
    ```

2.  **Run the Clustering Algorithm:**
    To perform the clustering and generate the visualization, run the `clustering.py` script.
    ```bash
    python clustering.py
    ```
    This will execute the entire pipeline and save the output visualization as `netflix_clusters.png`. It will also print a sample of titles from each of the 5 clusters to the console.

## 📁 File Descriptions

* **`clustering.py`**: The main script that handles data preprocessing, TF-IDF vectorization, K-Means clustering, and visualization.
* **`eda.py`**: A script for performing initial exploratory data analysis on the dataset.
* **`netflix_titles.csv`**: The raw dataset containing Netflix content information.
* **`requirements.txt`**: A list of the Python packages required to run the project.
* **`netflix_clusters.png`**: The output image file showing the visualized clusters.
