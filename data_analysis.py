import streamlit as st
import pandas as pd
import os
import hashlib

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from dtreeviz.trees import dtreeviz
from wordcloud import WordCloud

@st.cache_data(ttl=3600, show_spinner=True)
def __load_data(name):
    """Loads data. Private, exists to optimise caching"""
    return pd.read_csv(f"Data/{name}.csv")


@st.cache_data(ttl=3600, show_spinner=True)
def load_data(name, categorical=True):
    """Loads a given CSV by name from the ./Data folder. Optionally converts to categorical"""
    df = __load_data(name)
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

@st.cache_data(ttl=3600, show_spinner=True)
def train_decision_tree(df, target, max_depth=5):
    """
    Trains a decision tree on the given dataframe
    """
    model = DecisionTreeClassifier(max_depth=max_depth)
    model = model.fit(df.drop(target, axis=1).values, df[target].values)
    return model    

# ---- Utility Visualisation Functions ---- #
@st.cache_data(ttl=3600, show_spinner=True)
def visualize_decitree(_model, df, target, readable_df=None, max_depth=5):
    """
    Visualises a decision tree
    """
    if readable_df is None:
        readable_df = df
    
    strfix = lambda s: s.replace("_", " ").title()
    feature_names = list(map(strfix, readable_df.drop(target, axis=1).columns))
    class_names = list(map(strfix, readable_df[target].unique().astype(str)))
    
    viz = dtreeviz(_model, df.drop(target, axis=1), df[target], target_name=target, 
               feature_names=feature_names, 
               class_names=class_names)
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

df = load_data("SpotifyFeatures", categorical=False)
dataset_hash = hashlib.md5(df.to_csv().encode()).hexdigest()
if not hash_matches_saved(dataset_hash):
    # Save the hash
    with open("Data/hash.txt", "w") as f:
        f.write(dataset_hash)
        
    df_cat = load_data("SpotifyFeatures", categorical=True)
    
    print("Fitting with LASSO")
    model = make_pipeline(StandardScaler(), Lasso(alpha=0.8, random_state=42, warm_start=True))
    model.fit(df_cat.drop('popularity', axis=1), df_cat['popularity'])
    
    print("Fitting with Random Forest. May take up to 3min")
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=10, random_state=42, warm_start=True))
    model.fit(df.drop('popularity', axis=1), df['popularity'])
    
    # Create a dataframe of feature importance
    feature_importance = pd.DataFrame({
        'Feature': df.drop('popularity', axis=1).columns,
        'LASSO_Importance': model.steps[1][1].coef_,
        'RandomForest_Importance': model.steps[1][1].feature_importances_
    })
    
    # Add a column for the average importance, standardizing each column before averaging
    standard_rf_importance = (feature_importance['RandomForest_Importance'] - feature_importance['RandomForest_Importance'].mean()) / feature_importance['RandomForest_Importance'].std()
    standard_lasso_importance = (feature_importance['LASSO_Importance'] - feature_importance['LASSO_Importance'].mean()) / feature_importance['LASSO_Importance'].std()
    feature_importance['avg_importance'] = abs(standard_rf_importance + standard_lasso_importance) / 2
    feature_importance = feature_importance.sort_values('avg_importance', ascending=False)
    
    # Save the feature importance DB
    feature_importance.to_csv("Data/FeatureImportance.csv", index=False)