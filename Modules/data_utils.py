from streamlit import cache_data
import pandas as pd
import re

from sklearn.tree import DecisionTreeClassifier

import numpy as np
from pandasql import sqldf

from Modules.tree_util import prune_duplicate_leaves

def tostr(x):
    str_x = str(x)
    if isinstance(x, str): return x
    
    # If X is a number, return it with commas
    if re.match(r'^-?\d+$', str_x):
        return f"{int(str_x):,}"
    elif re.match(r'^-?\d+(?:\.\d+)?$', str_x):
        return f"{float(str_x):,}"
    
    # If X is a collection and not a string
    if hasattr(x, '__iter__') or hasattr(x, '__next__'):
        lst = list(map(tostr, x))
        if len(lst) == 0: return "None"
        elif len(lst) == 1: return lst[0]
        elif len(lst) == 2: return f"{lst[0]} and {lst[1]}"
        else: return f"{', '.join(lst[:-1])}, and {lst[-1]}"
    
    # If X is a boolean
    if isinstance(x, bool):
        if x: return "yes"
        else: return "no"
        
    # If X is None
    if x is None: return "None"
    
    # Otherwise, just return the string (we tried our best)
    return str_x

def train_test_split(df, frac=0.2):
    test = df.sample(frac=frac, axis=0, random_state=42)
    train = df.drop(index=test.index)
    return train, test

@cache_data(show_spinner=False)
def __load_data(name):
    """Loads data. Private, exists to optimize caching"""
    return pd.read_csv(f"Data/{name}.csv")

@cache_data(show_spinner=True)
def load_data(name, parse_categories=False):
    """Loads a given CSV by name from the ./Data folder. Optionally converts to categorical or prettifies strings"""
    df = __load_data(name)
    
    if parse_categories:
        # One-hot encode all categorical columns
        additional_columns = []
        columns_to_drop = []
        for col in df.columns:
            if df[col].dtype in (object, str):
                # If there are too many (>05%) unique values, don't one-hot encode
                if len(df[col].unique()) < 0.05 * len(df[col]):
                    additional_columns.append(pd.get_dummies(df[col], prefix=col))
                    columns_to_drop.append(col)
                else:
                    df[col] = pd.Categorical(df[col]).codes
                
        df = df.drop(columns_to_drop, axis=1)
        df = pd.concat([df] + additional_columns, axis=1)
                
    return df

@cache_data(show_spinner=False)
def prettify_data(df):
    # We want to shorten all strings to a more readable form
    # To do this, we do two things:
    # 1. replace all _ with " " and title case
    # 2. remove all words that are 3 characters or less IFF they make up leq 30% of the string
    # Step 2 causes a lot of collisions
    # To address this, we will keep track of transformations used
    # Stored as follows: Transformation (A->1) as (short_to_org[1] = A)
    # If a short is already in use, use str_ and add to to_revert
    short_to_org = dict()
    to_revert = set()
    def strfix(str_):
        str_ = str(str_).replace("_", " ").title().strip()
        shortstr = re.sub(r'(\s|^).{0,3}(?=\s|$)', '', str_).strip()
        
        if len(shortstr) > int(0.7 * len(str_)) and shortstr not in short_to_org:
            if str_ in short_to_org and short_to_org[str_] != str_:
                to_revert.add(str_)
                return str_
            short_to_org[shortstr] = str_
            return shortstr
        else:
            # If key already exists, we must revert existing keys
            if shortstr in short_to_org:
                to_revert.add(shortstr)
            return str_
        
    
    # Replace all strings in the dataframe with a prettier version, including the index
    for col in df.columns:
        if df[col].dtype in (object, str):
            df[col] = df[col].map(strfix)
    
    df = df.rename(columns=strfix)
    
    # Revert all collisions    
    def revert_collision(x):
        if x in to_revert:
            return short_to_org.get(x, x)
        return x
    
    for col in df.columns:
        if df[col].dtype in (object, str):
            df[col] = df[col].map(revert_collision)
    df = df.rename(columns=revert_collision)
    
    return df

def merge_onehot(df, column, features, aggregator='q3', remove=True):
    if isinstance(features, str):
        features = [features]
    
    rows_to_remove = list()
    for cat in features:
        cat = f"{cat}_"
        
        # Extract all rows beginnign w/ given cat
        cat_rows = [feat for feat in df[column] if str(feat).startswith(cat)]
        cat_rows = df[df[column].isin(cat_rows)]
        
        rows_to_remove += cat_rows.index.tolist()
        
        # Set all features in cat_rows to the cat name
        for row in cat_rows.index:
            cat_rows.at[row, column] = cat[:-1]
        
        # Calculate the median of cat_rows
        if   aggregator == 'q3': aggregator = lambda x: x.quantile(0.75)
        elif aggregator == 'q1': aggregator = lambda x: x.quantile(0.25)
        elif aggregator in ('mean', 'q2'): aggregator = np.mean
        elif aggregator == 'median': aggregator = np.median
        elif type(aggregator) == 'str':
            raise ValueError(f'Aggregator {aggregator} not recognized')
        
        cat_rows = cat_rows.groupby(column).agg({
            **{col: 'first' for col in df.columns},
            **{col: aggregator for col in df.columns if is_numeric(df[col])}  
        })
        
        # Append to the dataframe
        df = pd.concat([df, cat_rows], axis=0)
    
    if remove:
        df = df.drop(rows_to_remove, axis=0)
    
    return df    
    
@cache_data(show_spinner=False)
def __category_keys(df):
    """
    Returns a function that converts a categorical dataframe back to strings
    Private, exists to optimise caching
    """
    df = df.copy()
    categoricals = {}
    for col in df.columns:
        if df[col].dtype in (object, str):
            categoricals[col] = pd.Categorical(df[col])
    
    # Define & return a function that converts a dataframe back to strings
    def __convert_to_strings(df):
        df = df.copy()
        for col, cat in categoricals.items():
            df[col] = cat[df[col]]
        return df
    return __convert_to_strings

@cache_data(show_spinner=True)
def categorical_to_original(df):
    """
    Converts a categorical dataframe back to strings
    """
    return __category_keys(df)(df)

@cache_data(show_spinner=True)
def is_numeric(column):
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        return False
    
@cache_data(show_spinner=True)
def bin_series(series, bins):
    """
    Bins a series of data by bins by percentiles
    `bins` can be:
    - A dict of {bin_name: % of values}
    - A list of bin names
    In this case, bins will be evenly spaced
    - An int of the number of bins
    Again, bins will be evenly spaced
    
    All bins will be assigned low-to-high (eg. first bin will be lowest values)
    """
    if isinstance(bins, int):
        bins = list(f"Bin {x}" for x in range(0, bins))
    if isinstance(bins, list):
        count = 0
        bins = {bin: 1/len(bins) for bin in bins}
    if not isinstance(bins, dict):
        raise TypeError("bins must be int, dict or list")
    
    # Quick normalize
    sum_bins = sum(bins.values())
    bins = {bin: percent/sum_bins for bin, percent in bins.items()}
    
    # Make the bins monotonic
    bins = {bin: sum(list(bins.values())[:i+1]) for i, bin in enumerate(bins)}
    
    # Quick bin clamp (in case of float round error 1.00000001s)
    bins = {bin: min(1, percent) for bin, percent in bins.items()}
        
    # Pad bins with 0 and 1
    if 0 not in bins.values():
        percentiles = [0] + list(bins.values())
        labels = list(bins.keys())
    else:
        percentiles = list(bins.values())
        labels = list(bins.keys())[1:]
    
    # Divide the series into bins
    return pd.cut(series, bins=series.quantile(percentiles).values, labels=labels, include_lowest=True, duplicates="drop")

@cache_data(show_spinner=False)
def bin_data(df, columns:list, granularity=10):
    """Will bin the given rows of the dataframe into bins of size granularity across the entire dataframe's range.
    The resulting dataframe will have only the input columns and a new column: count,
    which is the number of rows in the original dataframe that fell into that bin."""
    
    df = df.copy()
    
    # Validate granularity
    if not isinstance(granularity, list):
        granularity = [granularity] * len(columns)
    
    # Generate the bin query
    bin_query = "".join(
        f"ROUND({col} / {gran}) * {gran} AS {col}, " for col, gran in zip(columns, granularity)
    )
    group_query = ", ".join(
        f"ROUND({col} / {gran}) * {gran}" for col, gran in zip(columns, granularity)
    )
    
    return sqldf(f"""
    SELECT 
        {bin_query}
        COUNT(*) AS count
    FROM df
    GROUP BY {group_query}
    """)
    
@cache_data(show_spinner=True)
def train_decision_tree(df, target, prune=False, **kwargs):
    """
    Trains a decision tree on the given dataframe
    """
    model = DecisionTreeClassifier(random_state=42, **kwargs)
    model = model.fit(df.drop(target, axis=1).values, df[target].values)
    if prune: prune_duplicate_leaves(model)
    return model