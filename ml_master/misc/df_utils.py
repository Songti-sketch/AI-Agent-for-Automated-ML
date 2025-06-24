import pandas as pd
import numpy as np

def df_to_str(df: pd.DataFrame) -> str:
    """Convert a pandas dataframe to a string representation.
    
    Sample Output:
    Make, Model, Year, Price, Mileage, Color
    Toyota, Corolla, 2015, 15000, 50000, Red
    Honda, Civic, 2018, 20000, 30000, Blue
    Ford, Focus, 2017, 18000, 40000, Green
    
    """
    # Convert the dataframe to a string representation
    df_str = df.to_string(index=False)
    # Add the column names to the string representation
    return df_str

def _get_distinct_values_of_column(df_column) -> str:
    """Print the distinct values of a pandas dataframe."""
    distinct_val = set(
        [str(i) for i in df_column]
    )
    return distinct_val

def get_distinct_values_all_columns(df: pd.DataFrame, max_num=40) -> str:
    distinct_dict = {}
    for col in df.columns:
        # first check the type of the column
        if df[col].dtype != "object":
            # simply pass
            continue
        # get the distinct values
        distinct_val = set(
            [str(i) for i in df[col]]
        )
        if len(distinct_val) > max_num:
            continue # also pass
        # add to the dictionary
        distinct_dict[col] = distinct_val

    ret = ""
    for col in distinct_dict:
        ret += f"The distinct values of the __{col}__ column are:\n: {distinct_dict[col]}\n"
    return ret

def get_continuous_columns_stat(df: pd.DataFrame) -> str:
    """Get the statistics of continuous columns in a pandas dataframe."""
    continuous_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {}
    for col in continuous_columns:
        # get number of NAs
        num_na = df[col].isna().sum()
        # get the min and max values
        min_val = df[col].min()
        max_val = df[col].max()
        # get the mean and std values
        mean_val = df[col].mean()
        std_val = df[col].std()
        # add to the dictionary
        stats[col] = {
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "num_na": num_na
        }
    ret = ""
    for col in stats:
        ret += f"The statistics of the __{col}__ column are:\n"
        ret += f"Min: {stats[col]['min']}\n"
        ret += f"Max: {stats[col]['max']}\n"
        ret += f"Mean: {stats[col]['mean']}\n"
        ret += f"Std: {stats[col]['std']}\n"
        ret += f"Num NA: {stats[col]['num_na']}\n"
    return ret

def parse_df_to_desc(df: pd.DataFrame) -> str:
    if len(df) < 5:
        df_str = df_to_str(df)
    else:
        df_str = df_to_str(df[:5])
    
    return (
        f"The data provided is a pandas dataframe. The columns are:\n"
        f"{df.columns.tolist()}\n"
        f"The data is:\n"
        f"{df_str}\n"
        f"The column types are:\n"
        f"{df.dtypes.tolist()}\n"
        f"{get_continuous_columns_stat(df)}\n"
        f"{get_distinct_values_all_columns(df)}"
    )

def parse_prob_to_desc(df: pd.DataFrame, target_column: str, metric_name: str):
    """Parse the problem to a string description.
    """
    df_desc = parse_df_to_desc(df)
    return (
        f"{df_desc}\n"
        f"The target column to predict is: {target_column}\n"
        f"The metric we need to evaluate the model is: {metric_name}\n"
    )