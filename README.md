
# Analysing Possible Predictors for a Song's Popularity

Team members:
Seung Won Seo, [Leon Leibmann](https://leibmann.org), Emelie Kyes, Oscar Wang

The currently hosted and deployed! Visit it here: [spotify-data.leibmann.org](https://spotify-data.streamlit.app/)

# Table of Contents

- [Analysing Possible Predictors for a Song's Popularity](#analysing-possible-predictors-for-a-songs-popularity)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Technologies](#technologies)
- [Data](#data)
- [Project Overview](#project-overview)
- [Setup](#setup)

# Overview

This webpage uses open-source Spotify data to predict a song's success. It extracts relevant features and develops models based on metrics such as popularity and play count. Users can interact with the dataset and experiment with different models. The project aims to help musicians and music enthusiasts understand the factors that contribute to a song's success.


We used an ensemble of methods


# Technologies

To create this, we used:

- [Streamlit](https://streamlit.io)
- [Matplotlib](https://matplotlib.org)
- [SeaBorn](https://seaborn.pydata.org)
- [Pandas](https://pandas.pydata.org)
- [WordCloud](https://amueller.github.io/word_cloud/)
- [PandasSQL](https://pypi.org/project/pandasql/)
- Elbow Grease 💪

# Data

The analysis is based on the [*Spotify Tracks DB*](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db), a dataset comprising of 232,725 randomly sample songs collected in 2019. It includes spotify's calculated features such as acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, and a song's overall popularity. This metric, reflecting a song's relative popularity on Spotify compared to others in its genre, is a close reflection of a song's actual popularity and is weighed heavily in spotify's song selection algorithm.

Keep in mind that the dataset captures a snapshot of the Spotify algorithm in 2019, and certain observations may no longer apply due to evolving metrics and algorithms.

# Project Overview

We delved into the analysis using a combination of observational methods and machine learning. Random forests and LASSO regression played a key role in identifying crucial features, with acousticness and loudness consistently standing out as influential factors across genres.

Genre-specific analyses provided nuanced insights into success metrics within each musical category. An interactive decision tree illustrated the intricacies of predicting popularity, particularly for elusive hit songs.

Examining independent artists revealed an intriguing observation. Smaller creators, despite having fewer songs, exhibited a talent for producing popular tracks—suggesting that quality outweighs quantity.

Expanding our focus to remixes, we compared original songs with their remixed counterparts, uncovering subtle changes that impact a track's popularity. The remixing process, regardless of genre, consistently influenced a song's features.

Notably, our machine learning model predicted the maximum potential improvement in popularity through strategic remixing. The average improvement across all songs reached an impressive 25%, highlighting the impact that small changes in a song during the mixing process can make.

# Setup

**Requirements.txt**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Poetry**
```bash
poetry install
poetry run streamlit run app.py
```

Make sure both streamlit and graphviz installed correctly. If not, try
again with conda or another build system.

The website should be available at [localhost:8501](http://localhost:8501)
