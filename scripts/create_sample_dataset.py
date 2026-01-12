"""
Script to create a sample dataset of 1000 rows from the full training dataset.

This script:
1. Loads the full features_train.pkl file
2. Samples 1000 rows (stratified if TARGET is available, otherwise random)
3. Saves the sample to features_train_sample_1000.pkl
4. Saves the client IDs to client_ids_sample_1000.json
"""
import pandas as pd
import pickle
import json
from pathlib import Path
import sys

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Paths
FEATURES_PATH = BASE_DIR / "src" / "dataset" / "features_train.pkl"
SAMPLE_FEATURES_PATH = BASE_DIR / "src" / "dataset" / "features_train_sample_1000.pkl"
CLIENT_IDS_PATH = BASE_DIR / "src" / "dataset" / "client_ids_sample_1000.json"
SAMPLE_SIZE = 1000
RANDOM_STATE = 42


def create_sample_dataset():
    """Create a sample dataset of 1000 rows"""
    print(f"📦 Loading full dataset from: {FEATURES_PATH}")
    
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {FEATURES_PATH}")
    
    # Load full dataset
    with open(FEATURES_PATH, 'rb') as f:
        df = pickle.load(f)
    
    print(f"✅ Dataset loaded: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check if TARGET column exists for stratified sampling
    if 'TARGET' in df.columns:
        print("📊 TARGET column found - using stratified sampling")
        # Stratified sampling
        df_sample = df.groupby('TARGET', group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), SAMPLE_SIZE // 2), 
                random_state=RANDOM_STATE
            )
        )
        
        # If we don't have enough samples, fill with random samples
        if len(df_sample) < SAMPLE_SIZE:
            remaining = SAMPLE_SIZE - len(df_sample)
            df_remaining = df.drop(df_sample.index).sample(
                n=remaining, 
                random_state=RANDOM_STATE
            )
            df_sample = pd.concat([df_sample, df_remaining])
        
        # Ensure exactly SAMPLE_SIZE rows
        df_sample = df_sample.sample(n=min(SAMPLE_SIZE, len(df_sample)), random_state=RANDOM_STATE)
    else:
        print("📊 No TARGET column - using random sampling")
        # Random sampling
        df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    
    print(f"✅ Sample created: {df_sample.shape}")
    
    # Save sample dataset
    print(f"💾 Saving sample dataset to: {SAMPLE_FEATURES_PATH}")
    with open(SAMPLE_FEATURES_PATH, 'wb') as f:
        pickle.dump(df_sample, f)
    print(f"✅ Sample dataset saved")
    
    # Extract and save client IDs
    if 'SK_ID_CURR' in df_sample.columns:
        client_ids = sorted(df_sample['SK_ID_CURR'].unique().tolist())
    else:
        # If SK_ID_CURR is the index
        client_ids = sorted(df_sample.index.unique().tolist())
    
    print(f"💾 Saving client IDs to: {CLIENT_IDS_PATH}")
    with open(CLIENT_IDS_PATH, 'w') as f:
        json.dump({"client_ids": client_ids, "total": len(client_ids)}, f, indent=2)
    print(f"✅ Client IDs saved: {len(client_ids)} IDs")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Original dataset: {df.shape[0]} rows")
    print(f"Sample dataset: {df_sample.shape[0]} rows")
    print(f"Client IDs: {len(client_ids)}")
    if 'TARGET' in df_sample.columns:
        target_dist = df_sample['TARGET'].value_counts()
        print(f"Target distribution:")
        print(f"  - Class 0: {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df_sample)*100:.1f}%)")
        print(f"  - Class 1: {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df_sample)*100:.1f}%)")
    print("="*60)
    print("✅ Sample dataset creation completed successfully!")


if __name__ == "__main__":
    try:
        create_sample_dataset()
    except Exception as e:
        print(f"❌ Error creating sample dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
