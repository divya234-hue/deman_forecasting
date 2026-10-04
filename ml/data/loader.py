import pandas as pd
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.
    """
    return pd.read_csv(file_path)