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
imdb = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.
    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.
    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.
    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset
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
    data = data_preprocessing(40000)
    tfidf = TfidfVectorizer(stop_words='english')
    #merge_ratings_movies['plot_keywords'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data['keyWords'])
    #tfidf_matrix.shape

    #merge_ratings_movies.head(2)
    tfidf.get_feature_names()[5000:5010]

    #merge_ratings_movies = merge_ratings_movies.drop('timestamp', axis=1)
    # Import linear_kernel
    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    lin_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(data['title']).drop_duplicates()
    # getting the indexes
    idx1 = indices[indices ==movie_list[0]].index[0]
    idx2 = indices[indices ==movie_list[1]].index[0]
    idx3 = indices[indices ==movie_list[2]].index[0]
    #
    sim_scores1 = list(enumerate(lin_sim[idx1]))
    sim_scores2 = list(enumerate(lin_sim[idx2]))
    sim_scores3 = list(enumerate(lin_sim[idx3]))

    # Sort the movies based on the similarity scores
    sim_scores1 = sorted(sim_scores1, key=lambda x: x[1], reverse=True)
    sim_scores2 = sorted(sim_scores2, key=lambda x: x[1], reverse=True)
    sim_scores3 = sorted(sim_scores3, key=lambda x: x[1], reverse=True)


    # Get the scores of the 10 most similar movies
    sim_scores1 = sim_scores1[0:top_n]
    sim_scores2 = sim_scores2[0:top_n]
    sim_scores3 = sim_scores3[0:top_n]
    

    # Get the movie indices
    movie_indices1 = [i[0] for i in sim_scores1]
    movie_indices2 = [i[0] for i in sim_scores2]
    movie_indices3 = [i[0] for i in sim_scores3]

    
    list_of_movie_indexes = [idx1,idx2,idx3]
    movie_indices1 = [i for i in movie_indices1 if i not in list_of_movie_indexes]
    movie_indices2 = [i for i in movie_indices2 if i not in list_of_movie_indexes]
    movie_indices3 = [i for i in movie_indices3 if i not in list_of_movie_indexes]

         
    result=[data['title'].iloc[movie_indices1][1:6],data['title'].iloc[movie_indices2][1:4],data['title'].iloc[movie_indices3][1:3]]

    recommended_movies = []
    for i in list(pd.concat(result,axis=0)):
      #if i not in movie_list:
        recommended_movies.append(i)
    

    
# Return the top 10 most similar movies
    return recommended_movies