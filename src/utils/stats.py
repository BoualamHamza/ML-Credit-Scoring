# this file contains functions that are used to help with the data exploration and analysis

# function that takes a data frame and extract the dtypes , the percentage of nan values and the percentage of unique values for each column min, the max , 25% , 50% , 75% , max, mean, std, count, the number of values per column and retuns a dataframe where each column is a feature and the rows are the stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive statistics for each column in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with statistics for each feature
    """
    stats_dict = {
        'feature_name': df.columns,
        'dtypes': df.dtypes.values,
        'nan_percentage': df.isna().mean().values,
        'unique_percentage': (df.nunique() / len(df)).values,
        'unique_count': df.nunique().values,
        'count': df.count().values,
    }
    
    # Only calculate numeric statistics for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Create a mapping of column names to their numeric stats
    mean_dict = numeric_df.mean().to_dict()
    std_dict = numeric_df.std().to_dict()
    min_dict = numeric_df.min().to_dict()
    q25_dict = numeric_df.quantile(0.25).to_dict()
    q50_dict = numeric_df.quantile(0.50).to_dict()
    q75_dict = numeric_df.quantile(0.75).to_dict()
    max_dict = numeric_df.max().to_dict()
    
    # Add numeric stats for all columns (None for non-numeric)
    stats_dict['mean'] = [mean_dict.get(col, None) for col in df.columns]
    stats_dict['std'] = [std_dict.get(col, None) for col in df.columns]
    stats_dict['min'] = [min_dict.get(col, None) for col in df.columns]
    stats_dict['25%'] = [q25_dict.get(col, None) for col in df.columns]
    stats_dict['50%'] = [q50_dict.get(col, None) for col in df.columns]
    stats_dict['75%'] = [q75_dict.get(col, None) for col in df.columns]
    stats_dict['max'] = [max_dict.get(col, None) for col in df.columns]
    
    return pd.DataFrame(stats_dict)


def plot_distribution(df: pd.DataFrame, column: str) -> None:
    """
    Plot the distribution of a column in a DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_distribution_by_target(df: pd.DataFrame, column: str) -> None:
    """
    Plot the distribution of a column in a DataFrame by the target.
    
    Args:
        df: Input DataFrame
        column: Column name
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True, hue=df['TARGET'])
    plt.title(f'Distribution of {column} by TARGET')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot the correlation matrix of a DataFrame.
    
    Args:
        df: Input DataFrame
    """
    plt.figure(figsize=(15, 10))
    num_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix')
    plt.show()


def violin_plot(df: pd.DataFrame, column: str) -> None:
    """
    Plot the violin plot of a column in a DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df[column], y=df['TARGET'])
    plt.title(f'Violin Plot of {column} by TARGET')
    plt.show()


def box_plot(df: pd.DataFrame, column: str) -> None:
    """
    Plot the box plot of a column in a DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column], y=df['TARGET'])
    plt.title(f'Box Plot of {column} by TARGET')
    plt.show()


