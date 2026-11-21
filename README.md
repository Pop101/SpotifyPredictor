# SpotifyPredictor

An interactive analysis of which audio features are associated with song popularity. Explore the dataset, compare genres and inspect what different models pick up.

[![Open GitHub issues](https://img.shields.io/github/issues/Pop101/SpotifyPredictor?style=flat-square)](https://github.com/Pop101/SpotifyPredictor/issues)

![SpotifyPredictor project cover](.github/images/readme-hero.svg)

## Table of Contents

- [Overview](#overview)
- [Technologies](#technologies)
- [Data](#data)
- [Project Overview](#project-overview)
- [Setup](#setup)

## Overview

Built by Seung Won Seo, [Leon Leibmann](https://leibmann.org/), Emelie Kyes and Oscar Wang. The app combines exploratory charts, genre comparisons and model visualizations. It describes associations in the dataset, not what will make a future song successful.

## Technologies

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Altair](https://altair-viz.github.io/)
- [Graphviz](https://graphviz.org/)

## Data

The [Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) is a historical snapshot of tracks, audio features and a Spotify popularity score. Popularity is the dataset's label, not an independent measure of quality, and neither the catalog nor the score should be assumed current.

## Project Overview

Compare genre signatures and feature importance, walk through a decision tree, and inspect the sections on independent artists and remixes. The results are exploratory; the remix comparisons do not establish a causal improvement in popularity.

## Setup

The manifest targets Python 3.11 or later within 3.x. From a checkout with Poetry installed:

```sh
poetry install --no-root
poetry run streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). Some model diagrams need Graphviz. The [original hosted demo](https://spotify-data.streamlit.app/) redirected to sign-in on review; a clean setup wasn't tested.
