from ml.data.loader import load_dataset
from ml.feature_engineering.features import create_features
file_path = "data/sample/sales.csv"
df = load_dataset(file_path)
feature_df = create_features(df)
print("\nOriginal columns:")
print(df.columns.tolist())
print("\nFeature columns:")
print(feature_df.columns.tolist())
print("\nFeature data:")
print(feature_df.head())