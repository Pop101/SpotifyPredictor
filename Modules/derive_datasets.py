import pandas as pd
from Modules.data_utils import load_data

import numpy as np
from tqdm import tqdm
from pandasql import sqldf
import pickle
from Levenshtein import ratio
import re

from sklearn import preprocessing

def generate_all_datasets():
    generate_artist_info()
    generate_genre_info()
    generate_underrepresented_artists()
    find_remix_pairs()
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
    # 2. find all songs with the word "remix" in the name
    # 3. cross join the dataset with itself, on the condition that the song name's Levenshtein ratio is > 0.9
    
    remix = re.compile(re.escape('remix'), re.IGNORECASE)
    normalize = lambda x: re.sub(r'[([][^)\]]*[)\]]|[^A-z]', '', remix.sub('', str(x)).lower())
    nratio = lambda x, y: ratio(normalize(x), normalize(y))
        
    songs = load_data("SpotifyFeatures")[['genre', 'track_name', 'track_id']]
    songs_with_remix = songs[songs['track_name'].apply(lambda name: remix.search(str(name)) is not None)]
    
    # Cross join the dataset with itself, ensuring that genre remains the same
    joined_songs = songs_with_remix.merge(songs, on='genre', suffixes=('', '_original'))
    
    # Calculate the Levenshtein ratio
    joined_songs['levenshtein_ratio'] = joined_songs.apply(lambda row: nratio(row['track_name'], row['track_name_original']), axis=1)
    
    # Filter on the Levenshtein ratio
    joined_songs = joined_songs[joined_songs['levenshtein_ratio'] > 0.8]
    joined_songs = joined_songs[joined_songs['track_id'] != joined_songs['track_id_original']]
    
    # Cut out all unnecessary columns
    joined_songs = joined_songs[['genre', 'track_name', 'track_id', 'genre_original', 'track_name_original', 'track_id_original', 'levenshtein_ratio']]
    joined_songs.columns = ['remix_genre', 'remix_name', 'remix_id', 'original_genre', 'original_name', 'original_id', 'similarity']
    
    joined_songs.to_csv('Data/RemixPairs.csv', index=False)

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