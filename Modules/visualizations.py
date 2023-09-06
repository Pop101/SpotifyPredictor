from streamlit import cache_data
import pandas as pd
from pandasql import sqldf

from altair import Chart, X, Y, Axis, Size, Scale, Color, Tooltip, Column
from Modules.visualization_utils import generate_wordcloud
from Modules.data_utils import load_data, merge_onehot, prettify_data, bin_series
from Modules.data_utils import train_test_split, train_decision_tree
from Modules.visualization_utils import visualize_decitree


@cache_data(show_spinner=True)
def song_pop_vs_avg_artist_pop():
    # Binned scatterplot of song popularity vs avg artist popularity
    artist_info = load_data("ArtistInfo")
    song_info = load_data("SpotifyFeatures")
    chart_data = pd.merge(song_info, artist_info, on='artist_name')
    
    # Use sqldf to quickly bin the popularity and avg_popularity
    # TODO: consider making a nicer binning function
    chart_data = sqldf("""
    SELECT 
        ROUND(popularity / 10) * 10 AS popularity,
        ROUND(avg_popularity / 10) * 10 AS avg_popularity,
        COUNT(*) AS count
    FROM chart_data
    GROUP BY ROUND(popularity / 10) * 10, ROUND(avg_popularity / 10) * 10
    """)

    chart = Chart(chart_data).mark_circle().encode(
        x = X('avg_popularity', title='Artist Popularity', bin=True),
        y = Y('popularity', title='Song Popularity', bin=True),
        size = Size('count', title='Number of Songs'),
    ).properties(
        title="Affect on Song Popularity by Feature",
    ).configure_axis(
        grid=False
    )
    
    return chart

@cache_data(show_spinner=True)
def most_popular_genres():
    data = load_data("SpotifyFeatures")
    data = data.groupby("genre").mean(numeric_only=True)
    data = data.sort_values(by="popularity", ascending=False).reset_index()

    # Quick altair chart of most popular genres
    chart = Chart(data).mark_bar().encode(
        x=X("genre", sort="-y", title="Genre"),
        y=Y("popularity", title="Popularity"),
        color=Color("popularity", legend=None, scale=Scale(scheme="tealblues")),
        tooltip=["genre", "popularity"]
    ).properties(
        title="Average song popularity by genre"
    )
    
    return chart

@cache_data(show_spinner=True)
def genre_signature_comparison(genre):
    """Create a plot of the feature averages for the selected genre,
    faceted on popularity category"""
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
    chart = Chart(genre_data).mark_boxplot(extent="min-max").encode(
        x=X('Feature:N', title=None, axis=Axis(labelAngle=-45)),
        y=Y('mean(value):Q', title='Mean Value'),
        color=Color('Feature:N', legend=None),
        # yo why are multiple tooltips broken?
        # https://github.com/vega/vega-lite/issues/7918
        tooltip=[
            Tooltip('Feature:N', title='Feature', format='.2f'),
            Tooltip('mean(value):Q', title='Center Value', format='.2f'),
            Tooltip('iqr(value):Q', title='Spread', format='.2f'),
        ]
    ).properties(
        width=200,  
        height=200
    ).facet(
        column=Column('Popularity:N', title=None)
    ).properties(
        title=f'Average Feature Values for {genre} Songs, By Popularity',
    )
    return chart

@cache_data(show_spinner=True)
def feature_importance_comparison():
    # Simple bar chart of feature importance.
    ft_imp = load_data("FeatureImportance")
    ft_imp = ft_imp[ft_imp['genre'] == 'All']
    ft_imp.drop('genre', axis=1, inplace=True)
    ft_imp = merge_onehot(ft_imp, 'feature', ['genre', 'key', 'mode', 'time_signature'])
    ft_imp = prettify_data(ft_imp)

    chart = Chart(ft_imp).mark_bar().encode(
        y=Y("Feature:N", sort="-x", title="Feature"),
        x=X("Importance:Q", title="Importance", axis=Axis(labels=False)),
        color=Color("Importance", legend=None, scale=Scale(scheme="goldgreen")),
        tooltip=["Feature", "Importance"]
    ).properties(
        title="Affect on Song Popularity by Feature",
    )
    return chart

@cache_data(show_spinner=True)
def feature_importance_genre(genre, top_n=5):
    # Plot Feature Importance again, but for the selected genre & with onehot this time
    ft_imp = load_data("FeatureImportance")
    ft_imp = ft_imp[prettify_data(ft_imp)['Genre'] == genre]
    ft_imp.drop(['genre'], axis=1, inplace=True)
    ft_imp = merge_onehot(ft_imp, 'feature', ['genre'])
    ft_imp = ft_imp[ft_imp['feature'] != 'genre']
    ft_imp = merge_onehot(ft_imp, 'feature', ['key','mode','time_signature'])
    ft_imp = prettify_data(ft_imp)

    chart = Chart(ft_imp).mark_bar().encode(
        y=Y("Feature:N", sort="-x", title="Feature"),
        x=X("Importance:Q", title="Importance", axis=Axis(labels=False)),
        color=Color("Importance", legend=None, scale=Scale(scheme="goldgreen")),
        tooltip=["Feature", "Importance"]
    ).properties(
        title=f"Top Features for {genre} Songs",
    )
    return chart

@cache_data(show_spinner=True)
def train_and_visualize_decision_tree(genre, max_depth=4):
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
    d_tree = train_decision_tree(train, 'popularity', prune=True, max_depth=max_depth, criterion='entropy')
    viz_svg = visualize_decitree(d_tree, train, 'popularity', pruned_data_readable, cmap='YlGnBu')
    
    # Gauge Accuracy
    test_only_hits = test[test['popularity'] == 3]
    accuracy = d_tree.score(test.drop('popularity', axis=1), test['popularity'])
    accuracy_hits = d_tree.score(test_only_hits.drop('popularity', axis=1), test_only_hits['popularity'])
    
    return viz_svg, d_tree, (accuracy, accuracy_hits)

@cache_data(show_spinner=True)
def generate_genre_wordcloud():
    data = load_data("SpotifyFeatures", parse_categories=False)
    data = data.groupby('genre').agg({'popularity': 'mean'})
    word_weights = data['popularity'].to_dict()
    return generate_wordcloud(word_weights)

@cache_data(show_spinner=True)
def generate_feature_wordcloud():
    data = load_data("FeatureImportance")
    data['feature'] = data['feature'].str.replace('duration_ms', 'Duration')
    data['feature'] = data['feature'].apply(lambda x: x.replace('_',' ').title())
    word_weights = data.set_index('feature').to_dict()['avg_importance']
    return generate_wordcloud(word_weights)
    