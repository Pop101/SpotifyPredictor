
# Core Pkgs
import streamlit as st 
from streamlit_util import load_css, add_image, header, render_draggable
from data_analysis import load_data, categorical_to_original, train_decision_tree, train_test_split, bin_series
from data_analysis import generate_genre_wordcloud, generate_feature_wordcloud, visualize_decitree

# Data Anal Pkgs
import pandas as pd 
import numpy as np

# Vizz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import altair as alt
import charttheme
alt.themes.enable('cs')

# Setup style and title
st.set_page_config(
    page_title="Song Popularity Analysis",
    page_icon="📻", # Can also be an image. TODO make one
    initial_sidebar_state="expanded",
    menu_items={} # menu gets deleted
)

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

We will be using a linear combination of LASSO regression, a random forest classifier,
and principal component analysis to determine
feature significance. This combination allows for wholistic analysis of the data,
performing feature selection, dimensionality reduction, and classification between
the many features of the dataset.

Performing this analysis, we can see the following feature importance:
""")
# Simple bar chart of feature importance. Remove y-axis labels
ft_imp = load_data("FeatureImportance", pretty=True)
sns.set_theme(style="whitegrid")
sns.barplot(x="Importance", y="Feature", data=ft_imp, label="Importance", color="b")
plt.xticks([])
st.pyplot()

st.markdown("""
Note how Genre is the most important indicator of popularity. This is because
some genres are simply more in the public lens than many others.

However, we can also see that artists are fairly unimportant. This is an
interesting result, as it means that the popularity of a song is rather independant
from the established artist handle many musicians have. This could also be furthered
by Spotify itself, which frequently pushes "radios" of songs that are have similar
features but not necessarily the same artist.

Because genre seems to be the biggest single indicator,
let's examine the popularity of each genre:
""")
data = load_data("SpotifyFeatures")
data = data.groupby("genre").mean()
data = data.sort_values(by="popularity", ascending=False).reset_index()

# Quick altair chart of most popular genres
st.altair_chart(alt.Chart(data).mark_bar().encode(
    x=alt.X("genre", sort="-y", title="Genre"),
    y=alt.Y("popularity", title="Popularity"),
    color=alt.Color("popularity", legend=None, scale=alt.Scale(scheme="tealblues")),
    tooltip=["genre", "popularity"]
).properties(
    title="Average song popularity by genre"
), use_container_width=True, theme=None)

st.markdown("""
This chart shows that the most popular genres are Pop, Rap, and Rock.
This is not surprising, as these are simple the most played genres
across both Spotify and music in general.

Children's music is also surprisingly popular, but this is likely due to
the number of children able to access Spotify, which has grown sharply
in the recent decade.     
""")

add_image(generate_feature_wordcloud(), caption="Word cloud of feature importance")

# Create a Series of Bars for each feature
header("Genre Analysis")

st.markdown("""
Breaking down genres, we can examine the feature profile specific to each to
determine what makes a genre unique and what makes a song popular within that genre.

Note that hit songs are the top 15% of songs within their genre,
while unpopular ones are the bottom 25%. Looking at these
extremes will help us determine what differentiates good from great.
""")

# input select box for genre
genre = st.selectbox(
    'Select a genre',
    load_data("SpotifyFeatures", pretty=False)['genre'].unique()
)

# Create a plot of the feature averages for the selected genre,
# faceted on popularity category
genre_data = load_data("SpotifyFeatures", pretty=True)
genre_data = genre_data.where(genre_data['Genre'] == genre).dropna()
genre_data = genre_data.sort_values(by='Popularity')
genre_data['Popularity'] = bin_series(genre_data['Popularity'], {'Unpopular': 25, 'Popular': 60, 'Hits': 15})

# Normalize the loudness feature, setting all values from 0 to 1
genre_data['LoudnessDB'] = genre_data['Loudness']
genre_data['Loudness'] = genre_data['Loudness'].apply(lambda x: (x - genre_data['Loudness'].min()) / (genre_data['Loudness'].max() - genre_data['Loudness'].min()))

# For each feature, calculate the average for each popularity category
#genre_data = genre_data.groupby(['Popularity']).mean().reset_index()
genre_data = pd.melt(genre_data, id_vars=['Popularity'], value_vars=[
    'Acousticness', 'Danceability', 'Instrumentalness',
    'Liveness', 'Speechiness', 'Valence', 'Loudness'],
var_name='Feature')

# Add feature names column
chart = alt.Chart(genre_data).mark_bar().encode(
    x=alt.X('Feature:N', title=None, axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('mean(value):Q', title='Mean Value'),
    color=alt.Color('Feature:N', legend=None),
).properties(
    width=200,
    height=200
).facet(
    column=alt.Column('Popularity:N', title=None)
).properties(
    title=f'Average Feature Values for {genre}, By Popularity',
)

# Plot figure
st.altair_chart(chart,use_container_width=True, theme=None)

st.markdown("""
You can use this widget to examine the feature profile of each genre.
Note how it changes within each facet, as the feature profile
often determines the popularity of a song its genre.
""")


# Create an interactive decision tree
header("Interactive Decision Tree", element="h2")

st.markdown("""
To dive deeper into the data, we can use a decision tree,
a machine learning algorithm that uses a series of yes/no questions
to classify data. 
""")

pruned_data = load_data("SpotifyFeatures").drop(
    # These columns are not needed for the decision tree - they are illegible to a user
    ['artist_name', 'track_name', 'track_id', 'key', 'mode', 'time_signature', 'genre'],
    axis=1
).sort_values(by='popularity')

#  Bin the popularity scores
pruned_data_readable = pruned_data.copy()
pruned_data['popularity'] = bin_series(pruned_data['popularity'], {1: 15, 2: 60, 3: 15})
pruned_data_readable['popularity'] = pruned_data['popularity'].replace({1: "Unpopular", 2: "Popular", 3: "Very Popular"})

# Train and visualize the decision tree
train, test = train_test_split(pruned_data, 0.3)
d_tree = train_decision_tree(train, 'popularity', prune=True, max_depth=4)
viz_svg = visualize_decitree(d_tree, train, 'popularity', pruned_data_readable, cmap='YlGnBu')
render_draggable(viz_svg, zoom_factor=1.7, initial_position=('+230px', '225px'))

# Show the accuracy of the decision tree
test_only_hits = test[test['popularity'] == 3]
accuracy = d_tree.score(test.drop('popularity', axis=1), test['popularity'])
accuracy_hits = d_tree.score(test_only_hits.drop('popularity', axis=1), test_only_hits['popularity'])
st.markdown(f"""
The accuracy of the decision tree is **{accuracy:.2%}**, that is,
if your song falls within a popular leaf, it is **{accuracy:.2%}** likely to be popular.

This is fairly good, considering the simplicity of the decision tree, however,
it also attests to the difficulty of creating hit songs.
Among hit songs, the accuracy is **{accuracy_hits:.2%}**.
These songs are the most difficult to predict, as they occur within all genres
and with any combination of features, all while being extremely rare.

A more complex model, such as a neural network, could be used to improve
this accuracy, however, it would be much more difficult to interpret and explain.
""")

# Create a heatmap of feature correlations
header("Individual Feature Correlations")

st.markdown("""
We can break down the correlations between features to see how they interact.
We have omitted categorical data, such as genre and artist name, as these have no
numeric values associated and therefore do not contribute to the correlation matrix.
""")

sp_dat = load_data("SpotifyFeatures", categorical=True, pretty=True)
corr = sp_dat.corr()
mask = np.diag(np.ones(len(corr)))
sns.heatmap(corr, mask=mask, cmap='YlGnBu', annot=False, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": 0.6})
st.pyplot()

st.markdown("""
Most feature-feature correlations are extremely explainable.
Energy and loudness are extremely correlated, as are danceability and valence.
Likewise, liveness and speachiness are correlated, as most live performances involve
significant singing and may include short introductions or interludes.
Loudness and acousticness are negatively correlated; it is simply harder to be loud
with acoustic instruments.
""")