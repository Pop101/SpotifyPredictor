from data_analysis import load_data
import pandas as pd

# Fix some discrepancies in the data

# discrepancy #1: Reggae = Reggaeton, so rename Reggaeton to Reggae
data = load_data("SpotifyFeatures")
data['genre'] = data['genre'].replace("Reggaeton", "Reggae")

# discrepancy #2: There are two genres called "Children's Music" and "Children’s Music"
data['genre'] = data['genre'].replace("Children’s Music", "Children's Music")

# save the new dataset, deleting the old
data.to_csv("Data/SpotifyFeatures.csv", index=False)