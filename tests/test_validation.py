from ml.data.loader import load_dataset
from ml.data.validator import validate_dataset
file_path = "data/sample/sales.csv"

df = load_dataset(file_path)

is_valid, errors = validate_dataset(df)

print("Dataset valid:", is_valid)
print("Errors:", errors)