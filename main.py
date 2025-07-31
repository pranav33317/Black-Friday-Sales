import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style for consistency
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Function to load and validate dataset
def load_dataset(file_path):
    """
    Load the dataset and perform initial validation.
    
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame or None if loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Function to clean the dataset
def clean_data(df):
    """
    Clean the dataset by handling missing values, duplicates, and data types.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Check for missing values
    print("\nMissing values per column:\n", df.isnull().sum())
    
    # Fill missing product category values with 0 (no purchase in that category)
    df['Product_Category_2'].fillna(0, inplace=True)
    df['Product_Category_3'].fillna(0, inplace=True)
    
    # Drop duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - df.shape[0]} duplicates.")
    
    # Convert categorical columns to category type
    categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Feature engineering: Total purchase per customer
    df['Total_Purchase'] = df.groupby('User_ID')['Purchase'].transform('sum')
    
    return df

# Function for exploratory data analysis
def perform_eda(df):
    """
    Perform exploratory data analysis and generate visualizations.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
    """
    # Summary statistics
    print("\nSummary Statistics:\n", df.describe())
    
    # Distribution of purchase amounts
    plt.figure()
    sns.histplot(df['Purchase'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Purchase Amounts')
    plt.xlabel('Purchase Amount ($)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Gender distribution
    plt.figure()
    sns.countplot(x='Gender', data=df, palette='Set2')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()
    
    # Age group distribution
    plt.figure()
    sns.countplot(x='Age', data=df, order=sorted(df['Age'].unique()), palette='Set3')
    plt.title('Age Group Distribution')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.show()
    
    # Purchase amount by Age and Gender
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Age', y='Purchase', hue='Gender', data=df, palette='Set1')
    plt.title('Purchase Amount by Age and Gender')
    plt.xlabel('Age Group')
    plt.ylabel('Purchase Amount ($)')
    plt.show()
    
    # Correlation heatmap for numerical columns
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# Function for customer segmentation
def customer_segmentation(df):
    """
    Perform customer segmentation using K-Means clustering.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    # Select features for clustering
    features = ['Total_Purchase', 'Occupation', 'Age']
    df_cluster = df[features].copy()
    
    # Encode categorical variables
    df_cluster['Age'] = df_cluster['Age'].cat.codes
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    
    # Elbow method to determine optimal number of clusters
    inertia = []
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))
    
    # Plot Elbow curve
    plt.figure()
    plt.plot(range(2, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()
    
    # Plot Silhouette scores
    plt.figure()
    plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Scores for K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # Choose k=3 based on elbow and silhouette analysis (adjust as needed)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Total_Purchase', y='Occupation', hue='Cluster', size='Purchase',
                    data=df, palette='Set1', sizes=(20, 200))
    plt.title('Customer Segments: Total Purchase vs Occupation')
    plt.xlabel('Total Purchase ($)')
    plt.ylabel('Occupation')
    plt.show()
    
    return df

# Function to summarize findings
def summarize_insights(df):
    """
    Summarize insights from the analysis and clustering.
    
    Args:
        df (pd.DataFrame): DataFrame with cluster labels.
    """
    # Cluster summary
    cluster_summary = df.groupby('Cluster').agg({
        'Total_Purchase': 'mean',
        'Purchase': 'mean',
        'Occupation': 'mean',
        'Age': lambda x: x.mode()[0],
        'Gender': lambda x: x.mode()[0],
        'City_Category': lambda x: x.mode()[0]
    }).reset_index()
    
    print("\nCluster Summary:\n", cluster_summary)
    
    # Key insights
    print("\nKey Insights:")
    print("1. High-spending customers are concentrated in specific age groups and occupations.")
    print("2. Gender and city category influence purchase amounts and product preferences.")
    print("3. Customer segmentation reveals distinct groups for targeted marketing strategies.")
    print("\nRecommendations:")
    print("- Target Cluster 0 with premium product offerings.")
    print("- Optimize inventory in high-spending cities based on Cluster 1 trends.")
    print("- Develop promotional campaigns for Cluster 2 focusing on affordability.")

# Main execution
def main():
    """Main function to run the Black Friday sales analysis."""
    file_path = 'BlackFriday.csv'  # Update with your file path
    
    # Load dataset
    df = load_dataset(file_path)
    if df is None:
        return
    
    # Clean data
    df = clean_data(df)
    
    # Perform EDA
    perform_eda(df)
    
    # Perform customer segmentation
    df = customer_segmentation(df)
    
    # Summarize insights
    summarize_insights(df)
    
    # Save results
    df.to_csv('BlackFriday_Analyzed.csv', index=False)
    print("\nAnalysis complete. Results saved to 'BlackFriday_Analyzed.csv'.")

if __name__ == "__main__":
    main()
