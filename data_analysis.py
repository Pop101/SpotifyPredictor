import streamlit as st
import pandas as pd
import os
import hashlib
import re

import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import preprocessing

from dtreeviz.trees import dtreeviz
from wordcloud import WordCloud

from tree_util import prune_duplicate_leaves

def train_test_split(df, frac=0.2):
    test = df.sample(frac=frac, axis=0, random_state=42)
    train = df.drop(index=test.index)
    return train, test

@st.cache_data(show_spinner=False)
def __load_data(name):
    """Loads data. Private, exists to optimize caching"""
    return pd.read_csv(f"Data/{name}.csv")


@st.cache_data(show_spinner=True)
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

@st.cache_data(show_spinner=False)
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
        str_ = str_.replace("_", " ").title().strip()
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

@st.cache_data(show_spinner=False)
def merge_onehot(df, column, features, remove=True):
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
        float_aggregator = lambda x: x.quantile(0.75) # 'median'
        cat_rows = cat_rows.groupby(column).agg({
            **{col: 'first' for col in df.columns},
            **{col: float_aggregator for col in df.columns if is_numeric(df[col])}  
        })
        
        # Append to the dataframe
        df = pd.concat([df, cat_rows], axis=0)
    
    if remove:
        df = df.drop(rows_to_remove, axis=0)
    
    return df    
    
@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=True)
def categorical_to_original(df):
    """
    Converts a categorical dataframe back to strings
    """
    return __category_keys(df)(df)

@st.cache_data(show_spinner=True)
def is_numeric(column):
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        return False
    
@st.cache_data(show_spinner=True)
def bin_series(series, bins):
    """
    Bins a series of data by percental bins
    `bins` can be a dict of {bin_name: percent} or a list of bin values
    Lists will be distributed evenly
    All bins will be assigned low-to-high (eg. first bin will be lowest values)
    """
    if isinstance(bins, list):
        bins = {bin: 1/len(bins) for bin in bins}
    if not isinstance(bins, dict):
        raise TypeError("bins must be a dict or list")
    
    # Quick normalize
    sum_bins = sum(bins.values())
    bins = {bin: percent/sum_bins for bin, percent in bins.items()}
    
    # Make the bins monotonic
    bins = {bin: sum(list(bins.values())[:i+1]) for i, bin in enumerate(bins)}
    
    # Pad bins with 0 and 1
    if 0 not in bins.values():
        percentiles = [0] + list(bins.values())
        labels = list(bins.keys())
    else:
        percentiles = list(bins.values())
        labels = list(bins.keys())[1:]
    
    # TODO: remove duplicate percentiles and match labels
    
    # Divide the series into bins
    return pd.cut(series, bins=series.quantile(percentiles).values, labels=labels, include_lowest=True, duplicates="drop")

@st.cache_data(show_spinner=True)
def train_decision_tree(df, target, prune=False, **kwargs):
    """
    Trains a decision tree on the given dataframe
    """
    model = DecisionTreeClassifier(random_state=42, **kwargs)
    model = model.fit(df.drop(target, axis=1).values, df[target].values)
    if prune: prune_duplicate_leaves(model)
    return model    

# ---- Utility Visualisation Functions ---- #
@st.cache_data(show_spinner=True)
def visualize_decitree(_model, df, target, readable_df=None, cmap="viridis"):
    """
    Visualises a decision tree
    """
    if readable_df is None:
        readable_df = prettify_data(df)
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    
    # Setup classes
    strfix = lambda s: s.replace("_ms","").replace("_", " ").title()
    feature_names = list(map(strfix, readable_df.drop(target, axis=1).columns))
    class_names = list(map(strfix, readable_df[target].unique().astype(str)))
    p_target_name = strfix(target)
    
    # Setup colors
    if 'matplotlib.colors' in str(type(cmap)).lower():
        # Draw colors from the colormap
        colors_list = cmap(np.linspace(0.05, 0.8, len(class_names)))
        cmap = [colors.rgb2hex(color) for color in colors_list]
    
    viz = dtreeviz(_model, df.drop(target, axis=1), df[target],
                target_name=p_target_name,
                feature_names=feature_names, 
                class_names=class_names,
                colors={'classes': [cmap] * (1+len(class_names))},)
    return viz.svg()


@st.cache_data(show_spinner=True)
def generate_wordcloud(input, width=800, height=400, colormap='YlGnBu'):
    if type(input) is str:
        w_freq = input.split()
    if type(input) is list:
        w_freq = {word: w_freq.get(word, 0) + 1 for word in input}
    if not 'w_freq' in locals():
        w_freq = input
        #for word in input: w_freq[word] = w_freq.get(word, 0) + 1
    
    wordcloud = WordCloud(
        width=width, height=height,
        background_color=None,
        mode='RGBA',
        colormap=colormap,
        max_words=50,
        prefer_horizontal=0.8,
        min_font_size=10,
        max_font_size=100,
        normalize_plurals=False,
        random_state=42,
        collocations=False,
        font_path="./Fonts/Helvetica.ttf"
    ).generate_from_frequencies(w_freq)
    
    wordcloud.recolor(color_func=None, random_state=42)
    
    # Return something that can easily be written to streamlit
    return wordcloud.to_image()


# --- Static, Specific Visualisations --- #
@st.cache_data(show_spinner=True)
def generate_genre_wordcloud():
    data = load_data("SpotifyFeatures", parse_categories=False)
    data = data.groupby('genre').mean()
    word_weights = data['popularity'].to_dict()
    return generate_wordcloud(word_weights)

@st.cache_data(show_spinner=True)
def generate_feature_wordcloud():
    data = load_data("FeatureImportance")
    data['feature'] = data['feature'].str.replace('duration_ms', 'Duration')
    data['feature'] = data['feature'].apply(lambda x: x.replace('_',' ').title())
    word_weights = data.set_index('feature').to_dict()['avg_importance']
    return generate_wordcloud(word_weights)
    

# On module load: Generate Feature Importance 
# if the dataset has changed at all

def hash_matches_saved(hash):
    if not os.path.exists("Data/hash.txt"): return False
    return open("Data/hash.txt").read() == hash

df = load_data("SpotifyFeatures", parse_categories=False)
dataset_hash = hashlib.new('md5')
dataset_hash.update(df.to_csv().encode('utf-8'))
dataset_hash.update(open("data_analysis.py").read().encode('utf-8'))
dataset_hash = dataset_hash.hexdigest()
if not hash_matches_saved(dataset_hash):
    df = load_data("SpotifyFeatures")
    df_cat = load_data("SpotifyFeatures", parse_categories=True)
    
    df = df.drop(['track_id', 'track_name'], axis=1)
    df_cat = df_cat.drop(['track_id', 'track_name'], axis=1)
    
    df_no_pop = df_cat.drop('popularity', axis=1)
    
    feature_importance =  pd.DataFrame({
        'feature': list(),
        'genre': list(),
        'lasso_importance': list(),
        'randomforest_importance': list(),
    })
    
    for genre in ['All'] + list(df['genre'].unique()):
        print(genre)
        
        # Filter on genre
        df_genre = df_cat[df_cat['genre_'+genre] == 1] if genre != 'All' else df_cat
        df_genre_no_pop = df_genre.drop('popularity', axis=1)
    
        print("Fitting with LASSO")
        lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.6, random_state=42, warm_start=True))
        lasso.fit(df_genre_no_pop, df_genre['popularity'])
    
        print("Fitting with Random Forest. May take up to 3min")
        forest = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=40, random_state=42, warm_start=True))
        forest.fit(df_genre_no_pop, df_genre['popularity'])
    
        # Create a dataframe of feature importance
        genre_ft_imp = pd.DataFrame({
            'feature': df_no_pop.columns,
            'genre': genre,
            'lasso_importance': lasso.steps[1][1].coef_,
            'randomforest_importance': forest.steps[1][1].feature_importances_,
        })
        
        # Gaussianize the feature importance
        for col in ['lasso_importance', 'randomforest_importance']:
            #genre_ft_imp[col] = preprocessing.scale(feature_importance[col])
            signs = np.sign(genre_ft_imp[col])
            genre_ft_imp[col] = np.abs(genre_ft_imp[col])
            genre_ft_imp[col] = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(genre_ft_imp[col].values.reshape(-1,1))
            genre_ft_imp[col] = genre_ft_imp[col] * signs
            
        # Make avg column
        genre_ft_imp['avg_importance'] = 2*abs(genre_ft_imp['randomforest_importance']) + abs(genre_ft_imp['lasso_importance'])
        genre_ft_imp['avg_importance'] /= 3
        
        # Append to the feature importance dataframe
        feature_importance = pd.concat([feature_importance, genre_ft_imp], axis=0)
    
    
    # Save the feature importance DB
    feature_importance.to_csv("Data/FeatureImportance.csv", index=False)
    
    # Now, attempt to remerge the results of the one-hot encodings
    
    # Save the hash
    with open("Data/hash.txt", "w") as f:
        f.write(dataset_hash)
        