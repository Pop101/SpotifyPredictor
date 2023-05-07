
# Core Pkgs
import streamlit as st 
from util import load_css, add_image, header

# Data Anal Pkgs
import pandas as pd 
from pandasql import sqldf
import numpy as np


# Vizz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# Setup style and title
st.set_page_config(
    page_title="Song Popularity Analysis",
    page_icon="📻", # Can also be an image. TODO make one
    initial_sidebar_state="expanded",
    menu_items={} # menu gets deleted
)
load_css("style-inject.css")

# Load the Data
st.cache_data(ttl=3600, show_spinner=True)
def load_data(name):
    df = pd.read_csv(f"Data/{name}.csv")
    return df

# Setup sidebar text
st.sidebar.markdown("""
# Navigation
""")
    
# The page itself
header("Song Popularity Analysis", element="h1")
add_image("Images/GenreCloud.png", caption="Word cloud of feature importance")

st.markdown(f"""
## Introduction

We are going to be analysing a dataset of songs from Spotify.
Containing {len(load_data("SpotifyFeatures"))} songs, we will be looking at the 
derived features of each song and how they affect the popularity of the song.
""")


header("Feature Importance", element="h2")

st.markdown("""

We will be using both a random forest classifier and LASSO regression to determine
feature significance.

Here is a simple bar char of the feature importance from the random forest classifier:
""")
# Simple bar chart of feature importance
ft_imp = load_data("FeatureImportance")
sns.set_theme(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x="rf_importance", y="feature", data=ft_imp)
st.pyplot()

st.markdown("""
This is a word cloud of the feature importance. I thought it looked cool,
and I *could*, therefore I *should*. Now I think it looks kinda bad tho.
""")

add_image("Images/FeatureCloud.png", caption="Word cloud of feature importance")

# Create a heatmap of feature correlations
header("Feature Correlations")

st.markdown("""
Here is a heatmap of the feature correlations.
""")

sp_dat = load_data("SpotifyFeatures")
sp_dat = sp_dat.drop(['track_id', 'track_name', 'artist_name', 'genre'], axis=1)
corr = sp_dat.corr()
mask = np.diag(np.ones(len(corr)))
sns.heatmap(corr, mask=mask, cmap='viridis', vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
st.pyplot()