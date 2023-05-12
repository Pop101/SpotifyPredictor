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

from dtreeviz.trees import dtreeviz
from wordcloud import WordCloud

from tree_util import prune_duplicate_leaves

def train_test_split(df, frac=0.2):
    test = df.sample(frac=frac, axis=0, random_state=42)
    train = df.drop(index=test.index)
    return train, test

@st.cache_data(ttl=3600, show_spinner=True)
def __load_data(name):
    """Loads data. Private, exists to optimise caching"""
    return pd.read_csv(f"Data/{name}.csv")


@st.cache_data(ttl=3600, show_spinner=True)
def load_data(name, categorical=True, pretty=False):
    """Loads a given CSV by name from the ./Data folder. Optionally converts to categorical or prettifies strings"""
    df = __load_data(name)
    
    if pretty:
        def strfix(str):
            str = str.replace("_", " ").title().strip()
            str_no_shorts = re.sub(r'(\s|^).{0,3}(?=\s|$)', '', str).strip()
            return str_no_shorts if len(str_no_shorts) > 0.5 * len(str) else str
        
        # Replace all strings in the dataframe with a prettier version, including the index
        for col in df.columns:
            if df[col].dtype in (object, str):
                df[col] = df[col].map(strfix)
        df = df.rename(columns=strfix)
    
    if not categorical:
        for col in df.columns:
            if df[col].dtype in (object, str):
                df[col] = pd.Categorical(df[col]).codes
                
    return df

@st.cache_data(ttl=3600, show_spinner=True)
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

@st.cache_data(ttl=3600, show_spinner=True)
def categorical_to_original(df):
    """
    Converts a categorical dataframe back to strings
    """
    return __category_keys(df)(df)

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

@st.cache_data(ttl=3600, show_spinner=True)
def train_decision_tree(df, target, prune=False, **kwargs):
    """
    Trains a decision tree on the given dataframe
    """
    model = DecisionTreeClassifier(random_state=42, **kwargs)
    model = model.fit(df.drop(target, axis=1).values, df[target].values)
    prune_duplicate_leaves(model)
    return model    

# ---- Utility Visualisation Functions ---- #
@st.cache_data(ttl=3600, show_spinner=True)
def visualize_decitree(_model, df, target, readable_df=None, cmap="viridis"):
    """
    Visualises a decision tree
    """
    if readable_df is None:
        readable_df = df
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


@st.cache_data(ttl=3600, show_spinner=True)
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
@st.cache_data(ttl=3600, show_spinner=True)
def generate_genre_wordcloud():
    data = load_data("SpotifyFeatures", categorical=True)
    data = data.groupby('genre').mean()
    word_weights = data['popularity'].to_dict()
    return generate_wordcloud(word_weights)

@st.cache_data(ttl=3600, show_spinner=True)
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

df = load_data("SpotifyFeatures", categorical=True)
dataset_hash = hashlib.new('md5')
dataset_hash.update(df.to_csv().encode('utf-8'))
dataset_hash.update(open("data_analysis.py").read().encode('utf-8'))
dataset_hash = dataset_hash.hexdigest()
if not hash_matches_saved(dataset_hash):
    df_cat = load_data("SpotifyFeatures", categorical=False)
    
    print("Fitting with LASSO")
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.8, random_state=42, warm_start=True))
    lasso.fit(df_cat.drop('popularity', axis=1), df_cat['popularity'])
    
    print("Fitting with Random Forest. May take up to 3min")
    forest = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=10, random_state=42, warm_start=True))
    forest.fit(df_cat.drop('popularity', axis=1), df_cat['popularity'])
    
    print("Fitting PCA importance")
    pca = make_pipeline(StandardScaler(), PCA(n_components=17))
    pca.fit(df_cat.drop('popularity', axis=1))
    
    # Create a dataframe of feature importance
    feature_importance = pd.DataFrame({
        'feature': df.drop('popularity', axis=1).columns,
        'lasso_importance': lasso.steps[1][1].coef_,
        'randomforest_importance': forest.steps[1][1].feature_importances_,
        'pcadims_importance': pca.steps[1][1].explained_variance_ratio_
    })
    
    # Add a column for the average importance, standardizing each column before averaging
    standard_rf_importance = (feature_importance['randomforest_importance'] - feature_importance['randomforest_importance'].mean()) / feature_importance['randomforest_importance'].std()
    standard_lasso_importance = (feature_importance['lasso_importance'] - feature_importance['lasso_importance'].mean()) / feature_importance['lasso_importance'].std()
    standard_pcadims_importance = (feature_importance['pcadims_importance'] - feature_importance['pcadims_importance'].mean()) / feature_importance['pcadims_importance'].std()
    
    feature_importance['avg_importance'] = (abs(standard_rf_importance) + abs(standard_lasso_importance) + abs(standard_pcadims_importance)) / 3
    feature_importance = feature_importance.sort_values('avg_importance', ascending=False)
    
    # Save the feature importance DB
    feature_importance.to_csv("Data/FeatureImportance.csv", index=False)
    
    # Save the hash
    with open("Data/hash.txt", "w") as f:
        f.write(dataset_hash)
        