import pandas as pd
from pandasql import sqldf

songs = pd.read_csv('Data/SpotifyFeatures.csv')

# Make a quick table of artist information
artist_info = sqldf("""
SELECT artist_name, genre, COUNT(*) AS num_songs, AVG(popularity) AS avg_popularity
FROM songs GROUP BY artist_name, genre
""")

artist_info.to_csv('Data/ArtistInfo.csv', index=False)

# Make a quick table of genre information
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


# Try to identify independent artists
# Identification method #1: lowest 50% of artists by number of songs in dataset

# select lowest 50% of artists within each genre
indep_1 = sqldf("""
WITH artist_counts AS (
    SELECT artist_name, genre, COUNT(*) AS num_songs
    FROM songs GROUP BY artist_name, genre
    ORDER BY num_songs ASC
),
artists_ranked AS (
    SELECT *, ROW_NUMBER() OVER(PARTITION BY genre ORDER BY num_songs ASC) AS artist_rank
    FROM artist_counts
),
artists_per_genre AS (
    SELECT genre, COUNT(*) AS num_artists
    FROM artists_ranked
    GROUP BY genre
)
SELECT artist_name, genre, num_songs FROM artists_ranked
JOIN artists_per_genre USING(genre)
WHERE artist_rank <= num_artists / 2
""")

print(indep_1)

# Method 2: artists with less than average number of songs
indep_2 = sqldf("""
WITH genre_info AS (
    SELECT genre, COUNT(*) AS genre_total_songs, COUNT(DISTINCT artist_name) AS genre_total_artists, 1.0 * COUNT(*) / COUNT(DISTINCT artist_name) AS avg_songs_per_artist
    FROM songs
    GROUP BY genre
)
SELECT artist_name, genre, COUNT(*) AS num_songs
FROM songs
JOIN genre_info USING(genre)
GROUP BY artist_name, genre
HAVING COUNT(*) < avg_songs_per_artist
""")

print(indep_2)

# Method 2 seems to include more artists. Let's save a csv with all songs
# by independent artists by Method 2's description

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

# Another Idea: Artists with an average popularity of less than half the genre's average popularity
