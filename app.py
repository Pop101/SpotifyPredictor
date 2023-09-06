import streamlit as st 

from Modules.streamlit_util import load_css, header, add_image, render_draggable
from Modules.data_utils import tostr, prettify_data, load_data

# Set matplotlib to correct mode
# Only affects seaborn & matplotlib
# Both currently unused
import matplotlib
matplotlib.use("Agg")

# Import chart theme
from Modules import chart_theme

# Import all charts
from Modules.visualizations import song_pop_vs_avg_artist_pop
from Modules.visualizations import most_popular_genres
from Modules.visualizations import genre_signature_comparison
from Modules.visualizations import feature_importance_comparison
from Modules.visualizations import feature_importance_genre
from Modules.visualizations import train_and_visualize_decision_tree
from Modules.visualizations import generate_genre_wordcloud
from Modules.visualizations import generate_feature_wordcloud


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
    
# Heading
header("Song Popularity Analysis", element="h1")
st.markdown("[Leon Leibmann](leibmann.org), Seung Won Seo, Emelie Kyes, Oscar Wang")
add_image(generate_genre_wordcloud())

# The Page Itself
st.markdown("""
## Introduction

Spotify is a music streaming service that has become a staple of the music industry,
being a first-choice for many music listeners. With over 500 million active users [[1]](https://www.demandsage.com/spotify-stats/),
and over 1,800,000 new songs uploaded monthly, it is increasingly important for independent
artists to understand the Spotify algorithm and how it contributes to a song's monthly streams
and overall popularity.

Through a combination of visualizations and machine learning techniques,
we will seek to answer the core question of what makes a song popular on Spotify.
""")

header("Data Providence") # Double check this is the right word
st.markdown(f"""
To conduct this analysis, we used the [*Spotify Tracks DB*](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db),
a random sampling of around 10,000 songs from each genre on Spotify gathered by Zaheen Hamidani in 2019.
Containing {tostr(len(load_data("SpotifyFeatures")))} songs, this dataset also includes all of 
Spotify's calculated features for each song, which are the result of several internal algorithms
the streaming service uses to match songs to listeners.

These features include:
- **Acousticness,** a value from 0.0 to 1.0 that indicates the degree of confidence that a track is acoustic, with 1.0 being highly confident.
- **Danceability,** a score from 0.0 to 1.0 that measures how suitable a track is for dancing based on musical elements such as tempo, rhythm, and beat strength.
- **Energy,** a perceptual measure of a track's intensity and activity, with a value from 0.0 to 1.0. High energy tracks are typically fast, loud, and noisy.
- **Instrumentalness,** a value from 0.0 to 1.0 that predicts the likelihood of a track containing no vocals. Values closer to 1.0 indicate a greater likelihood of no vocal content.
- **Liveness,** a value that indicates the probability of a track being performed live. Values above 0.8 strongly suggest that the track was performed live.
- **Loudness,** measured in decibels (dB), loudness represents the average volume of a track, with values typically ranging between -60 and 0 dB.
- **Speechiness,** a score from 0.0 to 1.0 that detects the presence of spoken words in a track, with higher values indicating a higher likelihood of speech-like content.
- **Valence,** a measure from 0.0 to 1.0 that describes the musical positiveness conveyed by a track. High valence tracks sound more positive, while low valence tracks sound more negative.

Finally, it also includes **Popularity,** the metric being analyzed in this project. 
This value is a score from 0 to 100 that indicates how popular a song is compared to similar songs
on Spotify. Garnered by both the total streams how recent those streams are, this value is
always changing but directly affects the number of impressions a song receives as well as its prevalence
among the many algorithm-created radios and playlists on Spotify [[2]](https://diymusician.cdbaby.com/music-career/spotify-algorithm/).
Thereby, the popularity rating directly correlates with a song's cultural relevance and its success.

Some important considerations to note throughout this report is that a song's popularity
metrics are constantly changing. This dataset is a snapshot in time of the Spotify algorithm,
its actions and insights, from 2019. As such, some of the observations and conclusions drawn 
from this analysis may no longer apply even at the time of report creation.
""")
# Some research also suggests that a popularity of 20+ is required for a song to be considered for the
# *Release Radar* and a popularity of 30+ is required for the *Discover Weekly* playlist [2] 

# FOR LATER:  Possible improvements could be to conduct a more recent sampling of spotify data,
# as some metrics, such as popularity, are incredibly time dependent

header("The Problem")
st.markdown("""
For struggling musicians, the Spotify algorithm is a black box, often antagonized for its
unexplained and seemingly arbitrary decisions. For such artists, it becomes
easy to blame the algorithm for their lack of success. Therefore, we seek to
provide explainability through analysis on how the Spotify algorithm affects
both big-names and lesser-known artists.
""")

st.altair_chart(song_pop_vs_avg_artist_pop(), use_container_width=True, theme=None)

st.markdown("""
This chart shows the relationship between a song's popularity and the average
popularity of other songs released by the same artist.

It is demonstrably difficult for smaller artists to gain traction in the music industry.
Smaller artists, those with a low average song popularity, find
that their songs often remain unpopular. In the chart, they have a much
smaller range of popularities across their releases. Artists with a 
below-30 average popularity have nearly no chance of creating song with
more than 50 popularity.

Meanwhile, larger artists  experience a much wider range of
success, with both extremely high (70+) popularity songs
and less popular songs (0-20).

This, however, is not to say that smaller artists cannot create hit songs.
""")

header("Analysis Methods")
st.markdown("""

To conduct this analysis, we used a combination of data visualization
and machine learning techniques. We will start with an observational
breakdown of feature differences between popular and unpopular songs,
then use a random forest and LASSO regression to determine the most
impactful features on song popularity.

For this analysis, consider a song to be very popular if it
lies within the top 15\% of songs in its genre, popular if it lies
within the top 60\%, and unpopular if it lies within the bottom 25\%.
This divides the dataset into three categories, which we will use
to determine the most impactful features on song popularity.

We chose the random forest algorithm because it is a simple, yet powerful
tool to categorize data while determining feature importance. It is also
easy to understand and visualize. This algorithm forms the rough basis for feature importance.
We then introduce granularity by using LASSO regression.
Because LASSO regression seeks to reduce the number of features used in a model,
it can be used to accurately discount unimportant features and thus highlight important ones.
""")

header("Observational Analysis", element="h2")

st.markdown("""
First, we will examine the differences between very popular, popular and unpopular songs
through a series of bar charts, scatter plots, and boxplots. It's critical to restate
that popularity is an ever-changing metric, and that this analysis is merely a snapshot.
However, by examining the differences between such songs, we can determine
differences between the songs that are popular and those that are not.
""")

# Quick altair chart of most popular genres
st.altair_chart(most_popular_genres(), use_container_width=True, theme=None)
data = load_data("SpotifyFeatures")

st.markdown(f"""
Because popularity is a relative metric, it is important to examine it
relative to other songs within its own genre. This chart shows the average
popularity of all songs of each genre. Note that the most popular genres
are {tostr(data['genre'][:3])}. This is not surprising, as these are simple the most played genres.

Now, lets focus in on each genre individually.
""")

# input select box for genre
# First, create a list of genres by avg. popularity
genres_by_pop = load_data("SpotifyFeatures").groupby("genre").mean(numeric_only=True)
genres_by_pop = genres_by_pop.sort_values(by="popularity", ascending=False).reset_index()
genres_by_pop = prettify_data(genres_by_pop)

# Now, allow user selection
genre = st.selectbox(
    'Select a genre',
    genres_by_pop['Genre'].unique(),
    index=list(genres_by_pop['Genre'].unique()).index('Pop')
)


# Plot figure
st.altair_chart(genre_signature_comparison(genre),use_container_width=True, theme=None)

st.markdown("""
You can use this widget to examine the feature profile of each genre.
Note how it changes within each facet, as the feature profile
often determines the popularity of a song its genre.
""")

header("Feature Importance", element="h2")

st.markdown("""
To gauge the importance of each feature, we train both a random forest classifier and
a LASSO regression model on the data. The random forest classifier is used to calculate
a feature's Gini importance, which is a measure of how well splits on that feature
decrease the impurity of the resulting nodes. This forms 66% of the final
feature importance score.

The LASSO regression model is used to further determine feature importances
by penalizing features whose weights are discarded by the model. This forms the final 
33% of the feature importance score.

Both models are trained with set seeds. The random forest classifier uses
40 individual estimators and the LASSO regression model uses a lambda of 0.6.
All categorical features are one-hot encoded and re-merged to attain an average
importance score for the category. This is done to prevent the model from
overfitting on a single category. The Artist Name feature contained too many unique values
to be one-hot encoded, so it was removed from the dataset.
""")

st.altair_chart(feature_importance_comparison(), use_container_width=True, theme=None)

st.markdown("""
It's very interesting to see that, across all genres, the most important features are
a song's acousticness and loudness.

It's also important to note that genre is within the top 3 most important features.
A lot of the time, a song's genre is a good indicator of its popularity,
as it determines the audience that will listen to it. This is especially true
for genres such as Children's Music, which is almost exclusively listened to by children
and thus rather unpopular overall. This can also be seen in "Pop" music, short for popular,
is defined as the most popular music of the time and is thus the most popular genre overall.

As a bonus, these feature importance scores make a very nice word cloud.
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

st.altair_chart(feature_importance_genre(genre), use_container_width=True, theme=None)


# Create an interactive decision tree
header("Interactive Decision Tree", element="h2")

st.markdown("""
To dive deeper into the data, we can use a decision tree,
a machine learning algorithm that uses a series of yes/no questions
to classify data. Note that this is an independently trained
decision tree, not from the random forest, with a max depth of 4.
This keeps the tree small and easy to understand, while still
being fairly accurate.
""")

viz_svg, _, (accuracy, accuracy_hits) = train_and_visualize_decision_tree(genre)
render_draggable(viz_svg, zoom_factor=1.7, initial_position=('250px', 0))

st.markdown(f"""
The accuracy of the decision tree is **{accuracy:.2%}**, that is,
if your song falls within a popular leaf, it is **{accuracy:.2%}** likely to be popular.

This is fairly good, considering the simplicity of the decision tree, however,
it also attests to the difficulty of creating hit songs.
Among hit songs, the accuracy is **{accuracy_hits:.2%}**.
These songs are the most difficult to predict, as they occur within all genres
and with any combination of features, all while being extremely rare.

A more complex model, such as a deeper tree, a random forest of multiple decision trees 
or a neural network, could be used to reduce bias and increase accuracy.
However, it would be much more difficult to interpret and explain.
""")

# header("Among Independent Artists")
# Reconduct feature importance selection, but remove both
# artist name and filter by artists whose average popularity is in the bottom 15%