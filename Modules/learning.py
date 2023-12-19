from Modules.data_utils import load_data

import pickle
import numpy as np
from tqdm import tqdm
from os import makedirs

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
    
def perform_ml(n_estimators=40, random_state=42):
    """Performs machine learning on the Spotify dataset to determine feature
    importance. Saves the resulting modesl in ./Models/<Genre>/[lasso|forest].pkl"""
    
    raw_data = load_data("SpotifyFeatures")
    data_onehot = load_data("SpotifyFeatures", parse_categories=True)
    
    raw_data = raw_data.drop(['track_id', 'track_name'], axis=1)
    data_onehot = data_onehot.drop(['track_id', 'track_name'], axis=1)

    for genre in tqdm(['All'] + list(raw_data['genre'].unique())):
        print(f"Current Genre: {genre}")
        makedirs(f"Models/{genre}", exist_ok=True)
        
        # Filter on genre
        df_genre = data_onehot[data_onehot['genre_'+genre] == 1] if genre != 'All' else data_onehot
        df_genre_no_pop = df_genre.drop('popularity', axis=1)
    
        print("\tFitting with LASSO")
        lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.6, random_state=random_state, warm_start=True))
        lasso.fit(df_genre_no_pop, df_genre['popularity'])
        pickle.dump(lasso, open(f"Models/{genre}/lasso.pkl", 'wb'))
    
        print("\tFitting with Random Forest. May take up to 3min")
        forest = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, warm_start=True))
        forest.fit(df_genre_no_pop, df_genre['popularity'])
        pickle.dump(forest, open(f"Models/{genre}/forest.pkl", 'wb'))

def calculate_max_change(model, input, max_delta, num_points=3):
    """Calculates the maximum change in model output that can occur 
    if the given input is changed by at most max_delta in all dimensions."""
    
    # We can't do gradient ascent because we don't have the gradient
    # Do random search instead
    
    dimensions = len(input)
    
    angles = np.random.uniform(0, 2 * np.pi, size=(num_points, dimensions - 1))
    distances = np.random.uniform(0, max_delta, size=num_points) ** (1 / (dimensions - 1))
    
    # Calculate points
    points = np.zeros((num_points, dimensions))
    points[:, 0] = distances
    
    for i in range(dimensions - 1):
        points[:, i + 1] = points[:, i] * np.sin(angles[:, i])
        points[:, i] = points[:, i] * np.cos(angles[:, i])

    points += input
    
    # Calculate max change
    current_output = model.predict(input)
    max_change = 0
    for i in range(num_points):
        output = model.predict(points[i])
        # TODO: check what datatype output is
        change = np.linalg.norm(output - current_output, ord=2)
        max_change = max(max_change, change)
    
    return max_change