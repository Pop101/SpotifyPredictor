
# Core Pkgs
import streamlit as st 
from streamlit_util import load_css, add_image, header, render_draggable
from data_analysis import load_data, categorical_to_original, train_decision_tree, train_test_split, bin_series, prettify_data
from data_analysis import generate_genre_wordcloud, generate_feature_wordcloud, visualize_decitree, merge_onehot

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
sns.set_theme(style="whitegrid")

# Setup style and title
try:
    st.set_page_config(
        page_title="Song Popularity Analysis",
        page_icon="📻", # Can also be an image. TODO make one
        initial_sidebar_state="expanded",
        menu_items={} # menu gets deleted
    )
except: pass

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

# Simple bar chart of feature importance.
ft_imp = load_data("FeatureImportance")
ft_imp = ft_imp[ft_imp['genre'] == 'All']
ft_imp.drop('genre', axis=1, inplace=True)
ft_imp = merge_onehot(ft_imp, 'feature', ['genre', 'key', 'mode', 'time_signature'])
ft_imp = prettify_data(ft_imp)

chart = alt.Chart(ft_imp).mark_bar().encode(
    y=alt.Y("Feature:N", sort="-x", title="Feature"),
    x=alt.X("Importance:Q", title="Importance", axis=alt.Axis(labels=False)),
    color=alt.Color("Importance", legend=None, scale=alt.Scale(scheme="goldgreen")),
    tooltip=["Feature", "Importance"]
).properties(
    title="Affect on Song Popularity by Feature",
)
st.altair_chart(chart, use_container_width=True, theme=None)

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
data = data.groupby("genre").mean(numeric_only=False)
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
header("Genre Analysis", element="h1")

st.markdown("""
Breaking down genres, we can examine the feature profile specific to each to
determine what makes a genre unique and what makes a song popular within that genre.

Note that hit songs are the top 15% of songs within their genre,
while unpopular ones are the bottom 25%. Looking at these
extremes will help us determine what differentiates good from great.
""")

# input select box for genre
# First, create a list of genres by avg. popularity
genres_by_pop = load_data("SpotifyFeatures").groupby("genre").mean(numeric_only=False)
genres_by_pop = genres_by_pop.sort_values(by="popularity", ascending=False).reset_index()
genres_by_pop = prettify_data(genres_by_pop)

# Now, allow user selection
genre = st.selectbox(
    'Select a genre',
    genres_by_pop['Genre'].unique()
)

# Create a plot of the feature averages for the selected genre,
# faceted on popularity category
@st.cache_data(show_spinner=True)
def gen_genredata_plot(genre):
    genre_data = load_data("SpotifyFeatures")
    genre_data = prettify_data(genre_data)
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

    # convert dtypes to string (dont ask me why)
    genre_data['Popularity'] = genre_data['Popularity'].astype(str)
    genre_data['Feature'] = genre_data['Feature'].astype(str)

    # Add feature names column
    chart = alt.Chart(genre_data).mark_boxplot(extent="min-max").encode(
        x=alt.X('Feature:N', title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('mean(value):Q', title='Mean Value'),
        color=alt.Color('Feature:N', legend=None),
        # yo why are multiple tooltips broken?
        # https://github.com/vega/vega-lite/issues/7918
        tooltip=[
            alt.Tooltip('Feature:N', title='Feature', format='.2f'),
            alt.Tooltip('mean(value):Q', title='Center Value', format='.2f'),
            alt.Tooltip('iqr(value):Q', title='Spread', format='.2f'),
        ]
    ).properties(
        width=200,  
        height=200
    ).facet(
        column=alt.Column('Popularity:N', title=None)
    ).properties(
        title=f'Average Feature Values for {genre} Songs, By Popularity',
    )
    return chart
    
# Plot figure
chart = gen_genredata_plot(genre)
st.altair_chart(chart,use_container_width=True, theme=None)

st.markdown("""
You can use this widget to examine the feature profile of each genre.
Note how it changes within each facet, as the feature profile
often determines the popularity of a song its genre.
""")

# Plot Feature Importance again, but for the selected genre & with onehot this time
top_n = 5
ft_imp = load_data("FeatureImportance")
ft_imp = ft_imp[prettify_data(ft_imp)['Genre'] == genre]
ft_imp.drop(['genre'], axis=1, inplace=True)
ft_imp = merge_onehot(ft_imp, 'feature', ['genre'])
ft_imp = ft_imp[ft_imp['feature'] != 'genre']
ft_imp = merge_onehot(ft_imp, 'feature', ['key','mode','time_signature'])
ft_imp = prettify_data(ft_imp)

chart = alt.Chart(ft_imp).mark_bar().encode(
    y=alt.Y("Feature:N", sort="-x", title="Feature"),
    x=alt.X("Importance:Q", title="Importance", axis=alt.Axis(labels=False)),
    color=alt.Color("Importance", legend=None, scale=alt.Scale(scheme="goldgreen")),
    tooltip=["Feature", "Importance"]
).properties(
    title=f"Top Features for {genre} Songs",
)
st.altair_chart(chart, use_container_width=True, theme=None)


# Create an interactive decision tree
header("Interactive Decision Tree", element="h2")

st.markdown("""
To dive deeper into the data, we can use a decision tree,
a machine learning algorithm that uses a series of yes/no questions
to classify data. 
""")

pruned_data = load_data("SpotifyFeatures", parse_categories=True).drop(
    # These columns are not needed for the decision tree - they are illegible to a user
    ['artist_name', 'track_name', 'track_id'],
    axis=1
)

pruned_data = pruned_data[pruned_data[f'genre_{genre}'] == 1]

#  Bin the popularity scores
pruned_data_readable = pruned_data.copy()
pruned_data['popularity'] = bin_series(pruned_data['popularity'], {1: 15, 2: 60, 3: 25})
pruned_data_readable['popularity'] = pruned_data['popularity'].replace({1: "Unpopular", 2: "Popular", 3: "Very Popular"})

# Train and visualize the decision tree
train, test = train_test_split(pruned_data, 0.1)
d_tree = train_decision_tree(train, 'popularity', prune=True, max_depth=4, criterion='entropy')
viz_svg = visualize_decitree(d_tree, train, 'popularity', pruned_data_readable, cmap='YlGnBu')
render_draggable(viz_svg, zoom_factor=1.7, initial_position=('250px', 0))

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

