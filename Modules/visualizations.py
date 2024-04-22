from streamlit import cache_data
import pandas as pd

from altair import Chart, X, Y, Axis, Size, Scale, Color, Tooltip, Column
import seaborn as sns
import numpy as np

from Modules.visualization_utils import generate_wordcloud
from Modules.data_utils import load_data, merge_onehot, prettify_data, bin_series, bin_data
from Modules.data_utils import train_test_split, train_decision_tree
from Modules.visualization_utils import visualize_decitree


@cache_data(show_spinner=True)
def song_pop_vs_avg_artist_pop():
    # Binned scatterplot of song popularity vs avg artist popularity
    artist_info = load_data("ArtistInfo")
    song_info = load_data("SpotifyFeatures")
    chart_data = pd.merge(song_info, artist_info, on='artist_name')
    
    # Use sqldf to quickly bin the popularity and avg_popularity
    chart_data = bin_data(chart_data, ['popularity', 'avg_popularity'])

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

#@cache_data(show_spinner=True)
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

    # Ensure no NaN and inf
    pruned_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    pruned_data.dropna(inplace=True)
    
    #  Bin the popularity scores
    pruned_data_readable = pruned_data.copy()
    pruned_data['popularity'] = bin_series(pruned_data['popularity'], {0: 25, 1: 60, 2: 15})
    pruned_data_readable['popularity'] = pruned_data['popularity'].replace({0: "Unpopular", 1: "Popular", 2: "Very Popular"})

    # Train and visualize the decision tree
    train, test = train_test_split(pruned_data, 0.1)
    d_tree = train_decision_tree(train, 'popularity', prune=True, max_depth=max_depth, criterion='entropy')
    viz_svg = visualize_decitree(d_tree, train, 'popularity', pruned_data_readable, cmap='YlGnBu')
    
    # Gauge Accuracy
    test_only_hits = test[test['popularity'] == 2]
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
    
def generate_independent_artist_popularity():
    # A side-by-side of the number of songs in the bottom 15%, 60%, and bottom 25% of popularity
    # among small artists vs all artists
    song_popularity = load_data("SpotifyFeatures")

    indep_popularity = load_data("IndepSongs")
    indep_popularity['popularity'] = pd.cut(indep_popularity['popularity'], bins=song_popularity['popularity'].quantile([0, 0.25, 0.85, 1]).values, labels=['Unpopular', 'Popular', 'Very Popular'], include_lowest=True, duplicates="drop")
    
    song_popularity['popularity'] = bin_series(song_popularity['popularity'], {1: 25, 2: 60, 3: 15})
    song_popularity['popularity'] = song_popularity['popularity'].replace({1: "Unpopular", 2: "Popular", 3: "Very Popular"})
    
    # Convert to counts
    indep_counts = indep_popularity['popularity'].value_counts()
    song_counts = song_popularity['popularity'].value_counts()
    
    # Merge the data
    indep_counts = pd.DataFrame({'category': ['Independent'] * len(indep_counts.index), 'popularity': indep_counts.index, 'count': indep_counts.values})
    indep_counts['count'] /= len(indep_popularity['popularity'])
    
    song_counts = pd.DataFrame({'category': ['All'] * len(indep_counts.index), 'popularity': song_counts.index, 'count': song_counts.values})
    song_counts['count'] /= len(song_popularity['popularity'])
    
    total_counts = pd.concat([indep_counts, song_counts])
    total_counts.columns = ['Category', 'Popularity', 'Percentage of Songs']
    
    # Stacked bar chart
    sns.set(rc={'figure.figsize': (11.7, 6.27)})
    sns.barplot(data=total_counts, x='Popularity', y='Percentage of Songs', hue='Category', order=['Unpopular', 'Popular', 'Very Popular'])

@cache_data(show_spinner=True)
def generate_remixpair_overview():
    remix_pairs = load_data("RemixPairs")
    
    # Plot X: popularity of remix, Y: popularity of original
    return Chart(remix_pairs).mark_circle().encode(
        x=X('popularity_original:Q', title='Original Popularity'),
        y=Y('popularity:Q', title='Remix Popularity'),
        tooltip=['remix_name:N', 'original_name:N', 'popularity:Q', 'popularity_original:Q']
    ).properties(
        title="Popularity of Remix vs Original"
    )

@cache_data(show_spinner=True)
def generate_remix_genre_changes():
    # boxplots of remix stat changes, grouped by genre
    
    remix_pairs = load_data("RemixPairs")
    remix_pairs.dropna(inplace=True)
    
    return Chart(remix_pairs).mark_boxplot().encode(
        x=X('genre:N', title='Genre'),
        y=Y('distance_between:Q', title='Statistic Changes'),
        color=Color('genre:N', legend=None, scale=Scale(scheme="category20")),
        tooltip=['genre:N', 'popularity_change:Q']
    ).properties(
        title="Statistics Changes of Remix vs Original, by Genre"
    )
    
#@cache_data(show_spinner=True)
def generate_remix_changes():
    # plot X: remix stat changes, Y: popularity changes
    
    remix_pairs = load_data("RemixPairs")
    remix_pairs['popularity_change'] = remix_pairs['popularity'] - remix_pairs['popularity_original']
    
    remix_pairs.dropna(inplace=True)
    
    # We could bin data, but it gets rid of genre (cool colors)
    #remix_pairs = bin_data(remix_pairs, ['popularity_change', 'distance_between'])
    
    return Chart(remix_pairs).mark_circle().encode(
        x=X('distance_between:Q', title='Statistic Changes'),
        y=Y('popularity_change:Q', title='Popularity Changes'),
        #size=Size('count:Q', title='Number of Songs'),
        color=Color('genre:N', scale=Scale(scheme="category20")),
        tooltip=['genre:N', 'popularity_change:Q']
    ).properties(
        title="Change in Statistics vs Change in Popularity"
    )

def generate_maximum_song_improvement():
    max_changes = load_data("MaxChanges")
    
    # Calculate percent improvement
    max_changes['max_popularity'] = max_changes['popularity'] + max_changes['max_improvement']
    max_changes['percent_improvement'] = max_changes['max_improvement'] / max_changes['popularity']
    max_changes.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_changes.dropna(inplace=True)
    
    # Drop over 100% improvements (outliers)
    # Note: this does skew the boxplot but makes it MUCH more legible
    max_changes = max_changes[max_changes['percent_improvement'] < 1]
    
    # Create boxplot
    return Chart(max_changes).mark_boxplot().encode(
        x=X('genre:N', title='Genre'),
        y=Y('percent_improvement:Q', title='Maximum Change', axis=Axis(format='%')),
        color=Color('genre:N', legend=None, scale=Scale(scheme="category20")),
        tooltip=['genre:N', 'max_improvement:Q']
    ).properties(
        title="Maximum Change in Statistics, by Genre"
    )

def calculate_average_improvement():
    max_changes = load_data("MaxChanges")
    
    # Calculate percent improvement
    max_changes['max_popularity'] = max_changes['popularity'] + max_changes['max_improvement']
    max_changes['percent_improvement'] = max_changes['max_improvement'] / max_changes['popularity']
    max_changes.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_changes.dropna(inplace=True)
    
    # Drop over 100% improvements (outliers)
    max_changes = max_changes[max_changes['percent_improvement'] < 1]
    
    return max_changes['percent_improvement'].mean()