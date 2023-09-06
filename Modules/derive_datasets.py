import pandas as pd
from Modules.data_utils import load_data
from pandasql import sqldf

import numpy as np

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

def generate_all_datasets():
    generate_artist_info()
    generate_genre_info()
    generate_underrepresented_artists()
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
    
    out_of = 1 + len(list(raw_data['genre'].unique()))
    print("Beginning machine learning:")
    for i, genre in enumerate(['All'] + list(raw_data['genre'].unique())):
        print(f"Current Genre: {genre} ({i+1}/{out_of})")
        
        # Filter on genre
        df_genre = data_onehot[data_onehot['genre_'+genre] == 1] if genre != 'All' else data_onehot
        df_genre_no_pop = df_genre.drop('popularity', axis=1)
    
        print("\tFitting with LASSO")
        lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.6, random_state=42, warm_start=True))
        lasso.fit(df_genre_no_pop, df_genre['popularity'])
    
        print("\tFitting with Random Forest. May take up to 3min")
        forest = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=40, random_state=42, warm_start=True))
        forest.fit(df_genre_no_pop, df_genre['popularity'])
    
        # Create a dataframe of feature importance
        genre_ft_imp = pd.DataFrame({
            'feature': onehot_features.columns,
            'genre': genre,
            'lasso_importance': lasso.steps[1][1].coef_,
            'randomforest_importance': forest.steps[1][1].feature_importances_,
        })
        
        # Gaussianize the feature importance
        for col in ['lasso_importance', 'randomforest_importance']:
            #genre_ft_imp[col] = preprocessing.scale(feature_importance[col])
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