import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def get_scaler(method):
    if method == 'StandardScaler':
        return StandardScaler()
    elif method == 'Min-Max Scaler':
        return MinMaxScaler()
    elif method == 'RobustScaler':
        return RobustScaler()
    else:
        raise ValueError("Unknown method")

def preprocess_data(df, x_cols, method):
    scaler = get_scaler(method)
    scaled_data = scaler.fit_transform(df[x_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=x_cols, index=df.index)
    return scaled_df

def compare_preprocessing(df, x_cols):
    """Return a dict of all scaled DataFrames for side-by-side comparison."""
    results = {}
    for method in ['StandardScaler', 'Min-Max Scaler', 'RobustScaler']:
        results[method] = preprocess_data(df, x_cols, method)
    return results
