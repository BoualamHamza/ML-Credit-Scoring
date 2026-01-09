"""
Feature Data Export Utility

This module provides functions to save and load processed feature data.
Use this after running feature engineering to persist the processed dataset.
Supports both CSV and pickle formats, and provides data splitting utilities.
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split


def save_features(df: pd.DataFrame, 
                  output_dir: str = "data/processed",
                  filename: str = "train_features.csv",
                  save_feature_names: bool = True) -> dict:
    """
    Save processed features to CSV and optionally save feature names.
    
    Args:
        df: DataFrame with processed features (should include TARGET and SK_ID_CURR)
        output_dir: Directory to save the files
        filename: Name of the CSV file
        save_feature_names: Whether to save feature names to JSON
    
    Returns:
        Dictionary with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the main dataframe
    csv_path = output_path / filename
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved features to: {csv_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Target distribution: {df['TARGET'].value_counts().to_dict()}")
    
    result = {"csv_path": str(csv_path)}
    
    # Save feature names
    if save_feature_names:
        exclude_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        json_path = output_path / "feature_names.json"
        with open(json_path, 'w') as f:
            json.dump({
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "exclude_cols": exclude_cols
            }, f, indent=2)
        print(f"✅ Saved feature names to: {json_path}")
        print(f"   Number of features: {len(feature_names)}")
        result["json_path"] = str(json_path)
    
    return result


def load_features(input_dir: str = "data/processed",
                  filename: str = "train_features.csv") -> pd.DataFrame:
    """
    Load processed features from CSV.
    
    Args:
        input_dir: Directory containing the saved files
        filename: Name of the CSV file
    
    Returns:
        DataFrame with processed features
    """
    csv_path = Path(input_dir) / filename
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Features not found at {csv_path}. "
            "Please run Feature_engineering.ipynb first."
        )
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded features from: {csv_path}")
    print(f"   Shape: {df.shape}")
    
    return df


def load_feature_names(input_dir: str = "data/processed") -> list:
    """
    Load feature names from JSON.
    
    Args:
        input_dir: Directory containing the saved files
    
    Returns:
        List of feature names
    """
    json_path = Path(input_dir) / "feature_names.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Feature names not found at {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data["feature_names"]


def save_features_pickle(df: pd.DataFrame, 
                         output_dir: str = "data/processed",
                         filename: str = "features_train.pkl",
                         save_feature_names: bool = True) -> dict:
    """
    Save processed features to pickle format (preserves dtypes better).
    
    Args:
        df: DataFrame with processed features (should include TARGET and SK_ID_CURR)
        output_dir: Directory to save the files
        filename: Name of the pickle file
        save_feature_names: Whether to save feature names to JSON
    
    Returns:
        Dictionary with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the main dataframe as pickle
    pkl_path = output_path / filename
    with open(pkl_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"✅ Saved features to: {pkl_path}")
    print(f"   Shape: {df.shape}")
    print(f"   File size: {pkl_path.stat().st_size / 1024 / 1024:.2f} MB")
    if 'TARGET' in df.columns:
        print(f"   Target distribution: {df['TARGET'].value_counts().to_dict()}")
    
    result = {"pkl_path": str(pkl_path)}
    
    # Save feature names
    if save_feature_names:
        exclude_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        json_path = output_path / "feature_names.json"
        with open(json_path, 'w') as f:
            json.dump({
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "exclude_cols": exclude_cols
            }, f, indent=2)
        print(f"✅ Saved feature names to: {json_path}")
        print(f"   Number of features: {len(feature_names)}")
        result["json_path"] = str(json_path)
    
    return result


def load_features_pickle(input_dir: str = "data/processed",
                         filename: str = "features_train.pkl") -> pd.DataFrame:
    """
    Load processed features from pickle format.
    
    Args:
        input_dir: Directory containing the saved files
        filename: Name of the pickle file
    
    Returns:
        DataFrame with processed features
    """
    pkl_path = Path(input_dir) / filename
    
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Features not found at {pkl_path}. "
            "Please run 02_feature_engineering.ipynb first."
        )
    
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"✅ Loaded features from: {pkl_path}")
    print(f"   Shape: {df.shape}")
    
    return df


def split_train_val_test(df: pd.DataFrame,
                         target_col: str = "TARGET",
                         train_ratio: float = 0.6,
                         val_ratio: float = 0.2,
                         test_ratio: float = 0.2,
                         random_state: int = 42,
                         exclude_cols: list = None) -> dict:
    """
    Split data into Train/Validation/Test sets with stratification.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        train_ratio: Proportion for training (default 0.6)
        val_ratio: Proportion for validation (default 0.2)
        test_ratio: Proportion for test (default 0.2)
        random_state: Random seed for reproducibility
        exclude_cols: Columns to exclude from features (in addition to target)
    
    Returns:
        Dictionary with:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - feature_names: List of feature column names
            - split_info: Summary of the split
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Default columns to exclude
    if exclude_cols is None:
        exclude_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in [target_col] + exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    
    # First split: separate test set
    temp_ratio = 1 - test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: separate train and validation from temp
    val_ratio_adjusted = val_ratio / temp_ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Prepare split info
    split_info = {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_ratio": len(X_train) / len(df),
        "val_ratio": len(X_val) / len(df),
        "test_ratio": len(X_test) / len(df),
        "n_features": len(feature_cols),
        "train_positive_rate": y_train.mean(),
        "val_positive_rate": y_val.mean(),
        "test_positive_rate": y_test.mean(),
        "random_state": random_state
    }
    
    print("=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Total samples:    {split_info['total_samples']:,}")
    print(f"  Training set:     {split_info['train_samples']:,} ({split_info['train_ratio']:.1%})")
    print(f"  Validation set:   {split_info['val_samples']:,} ({split_info['val_ratio']:.1%})")
    print(f"  Test set:         {split_info['test_samples']:,} ({split_info['test_ratio']:.1%})")
    print(f"  Number of features: {split_info['n_features']}")
    print(f"\nPositive rates (class=1):")
    print(f"  Train: {split_info['train_positive_rate']:.2%}")
    print(f"  Val:   {split_info['val_positive_rate']:.2%}")
    print(f"  Test:  {split_info['test_positive_rate']:.2%}")
    print("=" * 60)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_cols,
        "split_info": split_info
    }


def save_data_splits(splits: dict, output_dir: str = "data/processed") -> dict:
    """
    Save train/val/test splits to pickle files.
    
    Args:
        splits: Dictionary from split_train_val_test()
        output_dir: Directory to save files
    
    Returns:
        Dictionary with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits_to_save = {
        "X_train": splits["X_train"],
        "y_train": splits["y_train"],
        "X_val": splits["X_val"],
        "y_val": splits["y_val"],
        "X_test": splits["X_test"],
        "y_test": splits["y_test"],
        "feature_names": splits["feature_names"],
        "split_info": splits["split_info"]
    }
    
    pkl_path = output_path / "data_splits.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(splits_to_save, f)
    
    print(f"✅ Data splits saved to: {pkl_path}")
    
    return {"pkl_path": str(pkl_path)}


def load_data_splits(input_dir: str = "data/processed") -> dict:
    """
    Load saved train/val/test splits from pickle.
    
    Args:
        input_dir: Directory containing saved splits
    
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test, etc.
    """
    pkl_path = Path(input_dir) / "data_splits.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Data splits not found at {pkl_path}. "
            "Please run the data splitting step first."
        )
    
    with open(pkl_path, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"✅ Loaded data splits from: {pkl_path}")
    print(f"   Train: {len(splits['X_train']):,} samples")
    print(f"   Val:   {len(splits['X_val']):,} samples")
    print(f"   Test:  {len(splits['X_test']):,} samples")
    
    return splits
