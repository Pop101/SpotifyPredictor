
# Core Pkgs
import streamlit as st 
from streamlit_util import load_css, add_image, header, render_draggable
from data_analysis import load_data, categorical_to_original, train_decision_tree, visualize_decitree, train_test_split
from data_analysis import generate_genre_wordcloud, generate_feature_wordcloud

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
#st.set_page_config(
#    page_title="Song Popularity Analysis",
#    page_icon="📻", # Can also be an image. TODO make one
#    initial_sidebar_state="expanded",
#    menu_items={} # menu gets deleted
#)
load_css("style-inject.css")

# Setup sidebar text
st.sidebar.markdown("""
# Navigation
""")
    
# The page itself
header("Song Popularity Analysis", element="h1")
add_image(generate_genre_wordcloud())

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
ft_imp = load_data("FeatureImportance", pretty=True)
sns.set_theme(style="whitegrid")
sns.set_color_codes("pastel")
sns.barplot(x="Importance", y="Feature", data=ft_imp, label="Importance", color="b")
st.pyplot()

st.markdown("""
This is a word cloud of the feature importance. I thought it looked cool,
and I *could*, therefore I *should*. Now I think it looks kinda bad tho.
""")

add_image(generate_feature_wordcloud(), caption="Word cloud of feature importance")

# Create an interactive decision tree
header("Interactive Decision Tree", element="h2")
pruned_data = load_data("SpotifyFeatures").drop(
    # These columns are not needed for the decision tree - they are illegible to a user
    ['artist_name', 'track_name', 'track_id', 'key', 'mode', 'time_signature', 'genre'],
    axis=1
)

#  Bin the popularity scores
pruned_data_readable = pruned_data.copy()
pruned_data['popularity'] = pd.cut(pruned_data['popularity'], bins=[-1, 25, 70, 100], labels=[0, 1, 2])
pruned_data_readable['popularity'] = pd.cut(pruned_data_readable['popularity'], bins=[-1, 25, 70, 100], labels=['Unpopular', 'Popular', 'Hit'])

# Train and visualize the decision tree
train, test = train_test_split(pruned_data, 0.3)
d_tree = train_decision_tree(train, 'popularity', prune=True, max_depth=4)
viz_svg = visualize_decitree(d_tree, train, 'popularity', pruned_data_readable)
render_draggable(viz_svg, zoom_factor=1.7, initial_position=('+215px', '400px'))

# Show the accuracy of the decision tree
accuracy = d_tree.score(test.drop('popularity', axis=1), test['popularity'])
st.markdown(f"""
The accuracy of the decision tree is **{accuracy:.2%}**, that is,
if your song falls within a popular leaf, it is **{accuracy:.2%}** likely to be popular.
""")

# Create a heatmap of feature correlations
header("Feature Correlations")

st.markdown("""
Here is a heatmap of the feature correlations.
""")

sp_dat = load_data("SpotifyFeatures", categorical=False, pretty=True)
corr = sp_dat.corr()
mask = np.diag(np.ones(len(corr)))
sns.heatmap(corr, mask=mask, cmap='YlGnBu', vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
st.pyplot()