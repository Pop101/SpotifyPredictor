from Modules.data_utils import load_data

import pandas as pd
import numpy as np
import pickle
import re

from Levenshtein import ratio
from sklearn import preprocessing
from tqdm.auto import tqdm
from pandasql import sqldf

tqdm.pandas()
    
def generate_all_datasets():
    generate_artist_info()
    generate_genre_info()
    generate_underrepresented_artists()
    find_remix_pairs()
    calculate_remix_differences()
    generate_feature_importances()

def generate_artist_info():
    """Generates a table of artist information, including the number of songs,
    and the average popularity of their songs, as represented in the dataset.
    Saves it to Data/ArtistInfo.csv"""
    
    songs = load_data("SpotifyFeatures")
    
    artist_info = sqldf("""
        SELECT artist_name, genre, COUNT(*) AS num_songs, AVG(popularity) AS avg_popularity
        FROM songs GROUP BY artist_name, genre
    """)

    artist_info.to_csv('Data/ArtistInfo.csv', index=False)
    return artist_info

def generate_genre_info():
    """Generates a table of genre information, including the number of songs,
    their average popularity, the number of artists, and the average number of
    songs per artist within the genre. Saves it to Data/GenreInfo.csv"""
    
    songs = load_data("SpotifyFeatures")
    
    genre_info = sqldf("""
    SELECT genre,
        COUNT(*) AS total_songs,
        AVG(popularity) AS avg_popularity,
        COUNT(DISTINCT artist_name) AS total_artists,
        1.0 * COUNT(*) / COUNT(DISTINCT artist_name) AS avg_songs_per_artist
    FROM songs
    GROUP BY genre
    """)

    genre_info.to_csv('Data/GenreInfo.csv', index=False)
    return genre_info

def generate_underrepresented_artists():
    """Generates a table of songs by artists who have fewer songs than the
    genre's average number of songs per artist. Saves it to Data/IndepSongs.csv"""
    
    songs = load_data("SpotifyFeatures")
    
    indep_songs = sqldf("""
    WITH genre_info AS (
        SELECT genre, COUNT(*) AS genre_total_songs, COUNT(DISTINCT artist_name) AS genre_total_artists, 1.0 * COUNT(*) / COUNT(DISTINCT artist_name) AS avg_songs_per_artist
        FROM songs
        GROUP BY genre
    ),
    indep_artists AS (
        SELECT artist_name, genre, COUNT(*) AS num_songs
        FROM songs
        JOIN genre_info USING(genre)
        GROUP BY artist_name, genre
        HAVING COUNT(*) < avg_songs_per_artist
    )
    -- Exact same schema as original dataset
    SELECT genre,artist_name,track_name,track_id,popularity,acousticness,danceability,duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,time_signature,valence
    FROM songs
    JOIN indep_artists USING(artist_name, genre)
    """)

    # A more descriptive name would be underrepresented artists,
    # not independent artists
    indep_songs.to_csv('Data/IndepSongs.csv', index=False)
    return indep_songs

def find_remix_pairs():
    """Finds songs marked as remixes and the corresponding original song.
    Very conservative, only matches if the remix name is VERY similar"""
    
    # The process to find remixes will be as follows:
    # 1. load the dataset
    # 2. find all songs with the word "remix" or "live" in the name
    # 3. cross join the dataset with itself, on the condition that the song name's Levenshtein ratio is > 0.9
    # We now have a list of what is *probably* different versions of the same song
    
    remix = re.compile(r'(remix)|(live)', re.IGNORECASE)
    normalize = lambda x: re.sub(r'[([][^)\]]*[)\]]|[^A-z]', '', remix.sub('', str(x)).lower())
    nratio = lambda x, y: ratio(normalize(x), normalize(y))
        
    songs = load_data("SpotifyFeatures")[['genre', 'track_name', 'track_id']]
    songs_with_remix = songs[songs['track_name'].apply(lambda name: remix.search(str(name)) is not None)]
    
    # Cross join the dataset with itself, ensuring that genre remains the same
    joined_songs = songs_with_remix.merge(songs, on='genre', suffixes=('', '_original'))
    
    # Calculate the Levenshtein ratio
    print('Calculating Levenshtein ratio')
    joined_songs['levenshtein_ratio'] = joined_songs.progress_apply(lambda row: nratio(row['track_name'], row['track_name_original']), axis=1)
    
    # Filter on the Levenshtein ratio
    joined_songs = joined_songs[joined_songs['track_id'] != joined_songs['track_id_original']]
    joined_songs = joined_songs[joined_songs['levenshtein_ratio'] > 0.9]

    # Cut out all unnecessary columns
    joined_songs = joined_songs.reset_index(drop=True)
    joined_songs.columns = ['genre', 'remix_name', 'remix_id', 'original_name', 'original_id', 'similarity']
    
    joined_songs.to_csv('Data/RemixPairs.csv', index=False)

def calculate_remix_differences():
    """Uses found remix pairs to calculate the euclidean distance between
    the features of the remix and the original. Saves as a new column in Data/RemixPairs.csv
    Also adds remix_popularity and original_popularity columns to the dataset"""
    
    songs = load_data("SpotifyFeatures")
    remix_pairs = load_data("RemixPairs")
    initial_remix_columns = list(remix_pairs.columns)
    
    # Join the remix pairs with the original dataset
    remix_pairs = remix_pairs.merge(songs, left_on='remix_id', right_on='track_id', suffixes=('', '_remix'))
    remix_pairs = remix_pairs.merge(songs, left_on='original_id', right_on='track_id', suffixes=('', '_original'))
    
    # Calculate the euclidean distance between the features
    features_numeric = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
    features_categorical = ['key', 'mode', 'time_signature']
    features_categorical_factor = 0.3
    
    datapoints = list()
    for row in tqdm(remix_pairs.itertuples()):
        row = row._asdict()
        org_features, remix_features = list(), list()
        for feature in features_numeric:
            org_features += [row[f'{feature}_original']]
            remix_features += [row[f'{feature}']]
        
        for feature in features_categorical:
            # If they are the same, add 0 to both lists
            # otherwise, add 1 to the original and 0 to the remix
            if row[f'{feature}_original'] == row[f'{feature}']:
                org_features += [0]
                remix_features += [0]
            else:
                org_features += [features_categorical_factor]
                remix_features += [0]
        
        datapoints += [np.linalg.norm(np.array(org_features) - np.array(remix_features), ord=2)]
    
    remix_pairs = remix_pairs[initial_remix_columns + ['popularity', 'popularity_original']]
    remix_pairs['euclidean_distance'] = datapoints
    remix_pairs.to_csv('Data/RemixPairs.csv', index=False)

def generate_feature_importances():
    """Generates a table of feature importances for each genre, using LASSO and
    random forest classifications. Takes around 5-10 minutes. 
    Saves it to Data/FeatureImportance.csv"""
    
    raw_data = load_data("SpotifyFeatures")
    data_onehot = load_data("SpotifyFeatures", parse_categories=True)
    
    raw_data = raw_data.drop(['track_id', 'track_name'], axis=1)
    data_onehot = data_onehot.drop(['track_id', 'track_name'], axis=1)
    
    onehot_features = data_onehot.drop('popularity', axis=1)
    
    feature_importance =  pd.DataFrame({
        'feature': list(),
        'genre': list(),
        'lasso_importance': list(),
        'randomforest_importance': list(),
    })
    
    for genre in tqdm(['All'] + list(raw_data['genre'].unique())):        
        # Load Models
        lasso = pickle.load(open(f"Models/{genre}/lasso.pkl", 'rb'))
        forest = pickle.load(open(f"Models/{genre}/forest.pkl", 'rb'))
    
        # Create a dataframe of feature importance
        genre_ft_imp = pd.DataFrame({
            'feature': onehot_features.columns,
            'genre': genre,
            'lasso_importance': lasso.steps[1][1].coef_,
            'randomforest_importance': forest.steps[1][1].feature_importances_,
        })
        
        # Gaussianize the feature importance
        for col in ['lasso_importance', 'randomforest_importance']:
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
    return feature_importance