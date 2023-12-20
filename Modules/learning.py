from Modules.data_utils import load_data

import pickle
import numpy as np
from tqdm import tqdm
from os import makedirs

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore", module="sklearn")
    
def perform_ml(n_estimators=40, random_state=42):
    """Performs machine learning on the Spotify dataset to determine feature
    importance. Saves the resulting modesl in ./Models/<Genre>/[lasso|forest].pkl"""
    
    raw_data = load_data("SpotifyFeatures")
    data_onehot = load_data("SpotifyFeatures", parse_categories=True)
    
    raw_data = raw_data.drop(['track_id', 'track_name'], axis=1)
    data_onehot = data_onehot.drop(['track_id', 'track_name'], axis=1)

    for genre in (pbar := tqdm(['All'] + list(raw_data['genre'].unique()))):
        pbar.set_description(f"Genre {genre}")
        makedirs(f"Models/{genre}", exist_ok=True)
        
        # Filter on genre
        df_genre = data_onehot[data_onehot['genre_'+genre] == 1] if genre != 'All' else data_onehot
        df_genre_no_pop = df_genre.drop('popularity', axis=1)
    
        lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.6, random_state=random_state, warm_start=True))
        lasso.fit(df_genre_no_pop, df_genre['popularity'])
        pickle.dump(lasso, open(f"Models/{genre}/lasso.pkl", 'wb'))
    
        forest = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, warm_start=True))
        forest.fit(df_genre_no_pop, df_genre['popularity'])
        pickle.dump(forest, open(f"Models/{genre}/forest.pkl", 'wb'))

def calculate_max_change(model, input, max_delta, num_points=20, signed=False):
    """Calculates the maximum change in model output that can occur 
    if the given input is changed by at most max_delta in all dimensions."""
    
    # We can't do gradient ascent because we don't have the gradient
    # Do random search instead
    dimensions = len(input)
    
    points = np.random.rand(num_points, dimensions) * 2 - 1 # can use randn instead
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points *= max_delta
    points = [input + point for point in points]
    
    # Calculate max change
    original_output = model.predict(np.array(input).reshape(1, -1))
    max_change = -np.inf
    for i in range(num_points):
        output = model.predict(np.array(points[i]).reshape(1, -1))
        change = np.linalg.norm(output - original_output, ord=2)
        change = change if not signed else np.sign(np.average(output - original_output)) * change
        max_change = max(max_change, change)
    
    return max_change