"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
#ratings = pd.read_csv('resources/data/train.csv')
imdb = pd.read_csv('resources/data/imdb_data.csv')
movies.dropna(inplace=True)


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Function that takes in movie title as input and outputs most similar movies

    merge_ratings_movies = pd.merge(imdb, movies, on='movieId', how='inner')


    tfidf = TfidfVectorizer(stop_words='english')
    merge_ratings_movies['plot_keywords'].fillna('')
    tfidf_matrix = tfidf.fit_transform(merge_ratings_movies['plot_keywords'].values.astype('U'))
    tfidf_matrix.shape

    #merge_ratings_movies.head(2)
    tfidf.get_feature_names()[5000:5010]

    #merge_ratings_movies = merge_ratings_movies.drop('timestamp', axis=1)
    # Import linear_kernel
    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(merge_ratings_movies.index, index=merge_ratings_movies['title']).drop_duplicates()


    # Get the index of the movie that matches the title
    idx = indices[movie_list]



    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))


    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:top_n]
    

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    

    # Return the top 10 most similar movies
    return merge_ratings_movies['title'].iloc[movie_indices]
   