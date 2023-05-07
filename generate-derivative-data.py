from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
from wordcloud import WordCloud

def load_data(name, allow_strings=True):
    df = pd.read_csv(f"Data/{name}.csv")
    if not allow_strings:
        for col in df.columns:
            if df[col].dtype in (object, str):
                df[col] = pd.Categorical(df[col]).codes
    return df

def generate_feature_importance():
    # Model fitting function
    def calc_important_features_lasso():
        # Truncate the columsn that def don't matter
        df = load_data("SpotifyFeatures", allow_strings=False)
        
        # Fit a LASSO model to find which features are most important
        print("Fitting with LASSO")
        model = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=0.8, random_state=42, warm_start=True))
        model.fit(df.drop('popularity', axis=1), df['popularity'])
        
        # Return all feature's ranked importance
        df.drop('popularity', axis=1, inplace=True)
        return pd.DataFrame({'feature': df.columns, 'importance': model.steps[1][1].coef_}).sort_values('importance', ascending=False)

    def calc_important_features_randomforest():
        df = load_data("SpotifyFeatures", allow_strings=False)
        
        # Fit a random forest model to find which features are most important
        print("Fitting with Random Forest. Will take ca. 3min")
        model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=40, random_state=42, warm_start=True))
        model.fit(df.drop('popularity', axis=1), df['popularity'])
        
        # Return all feature's ranked importance
        df.drop('popularity', axis=1, inplace=True)
        return pd.DataFrame({'feature': df.columns, 'importance': model.steps[1][1].feature_importances_}).sort_values('importance', ascending=False)
    
    # Load the Data
    lasso_vals = calc_important_features_lasso()
    rf_vals = calc_important_features_randomforest()
    print("Done training!")
    
    # Merge the two dataframes
    lasso_vals.rename(columns={'importance': 'lasso_importance'}, inplace=True)
    rf_vals.rename(columns={'importance': 'rf_importance'}, inplace=True)
    merged = rf_vals.merge(lasso_vals, on='feature', how='inner')
    
    # Add a column for the average importance, standardizing each column before averaging
    standard_rf_importance = (merged['rf_importance'] - merged['rf_importance'].mean()) / merged['rf_importance'].std()
    standard_lasso_importance = (merged['lasso_importance'] - merged['lasso_importance'].mean()) / merged['lasso_importance'].std()
    merged['avg_importance'] = abs(standard_rf_importance + standard_lasso_importance) / 2
    
    # Save to CSV
    merged = merged.sort_values('avg_importance', ascending=False)
    merged.to_csv("Data/FeatureImportance.csv", index=False)

def generate_feature_word_cloud():
    ft_imp = pd.read_csv("Data/FeatureImportance.csv")
    # rename features to be human-readable
    ft_imp['feature'] = ft_imp['feature'].str.replace('duration_ms', 'Duration')
    ft_imp['feature'] = ft_imp['feature'].apply(lambda x: x.replace('_',' ').title())
    
    
    word_weights = ft_imp.set_index('feature').to_dict()['avg_importance']

    wordcloud = WordCloud(
        width=800, height=400,
        background_color=None,
        mode='RGBA',
        colormap='viridis',
        max_words=50,
        prefer_horizontal=0.8,
        min_font_size=10,
        max_font_size=100,
        normalize_plurals=False,
        random_state=42,
        # make words closer together
        collocations=False,
        font_path="./Images/Helvetica.ttf"
    ).generate_from_frequencies(word_weights)
    
    wordcloud.recolor(color_func=None, random_state=42)
    wordcloud.to_file("Images/FeatureCloud.png")

def generate_genre_word_cloud():
    sp_dat = pd.read_csv("Data/SpotifyFeatures.csv")
    
    # Create a weight dictionary based on average popularity of each genre
    sp_dat = sp_dat.groupby('genre').mean()
    word_weights = sp_dat['popularity'].to_dict()
    
    
    wordcloud = WordCloud(
        width=800, height=400,
        background_color=None,
        mode='RGBA',
        colormap='viridis',
        max_words=50,
        prefer_horizontal=0.8,
        min_font_size=10,
        max_font_size=100,
        normalize_plurals=False,
        random_state=42,
        font_path="./Images/Helvetica.ttf"
    ).generate_from_frequencies(word_weights)
    wordcloud.recolor(color_func=None, random_state=42)
    wordcloud.to_file("Images/GenreCloud.png")

print("Generating feature importance")
generate_feature_importance()

generate_feature_word_cloud()
generate_genre_word_cloud()