#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 03:15:10 2021

@author: rodrigopecchio
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# to print data
# print(movies.head())
# print(ratings.head())

# create new dataframe where each column represents each unique userID
# each row represents each unique movieId
# NaN (not a number) means there is no rating
final_dataset = ratings.pivot(index= 'movieId',
                              columns = 'userId',
                              values = 'rating')

# print(final_dataset.head())

# replace all NaN for 0 to clean dataset
final_dataset.fillna(0, inplace = True)
# print(final_dataset.head())

# noise reduction
# to qualify a movie, minimum 10 users should have voted a movie
# to qualify a user, minimum 50 movies should have voted by the user

# visualize these noise reduction filters
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# visualize number of users who voted with our threshold of 10
f,ax = plt.subplots(1, 1, figsize = (16, 4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index, no_user_voted, color = 'mediumseagreen')
plt.axhline(y = 10, color = 'r') # 10 user vote threshold
plt.xlabel('Movie ID')
plt.ylabel('Number of users who voted')
plt.show()

# make necessary modifications as per 10 vote threshold
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]

# visualize the number of votes by each user with 50 movie per user threshold
f,ax = plt.subplots(1, 1, figsize = (16, 4))
plt.scatter(no_movies_voted.index, no_movies_voted, color = 'mediumseagreen')
plt.axhline(y = 50, color = 'r')
plt.xlabel('UserId')
plt.ylabel('Number of votes by the user')
# plt.show()

# make necessary modifications as per 50 votes per user threshold
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]
# print(final_dataset)

# reduce sparsity of values to avoid huge computing when feeding to model

# how the csr_matrix function works
# sample = np.array([[0, 0, 3, 0, 0], [4, 0, 0, 0, 2], [0, 0, 0, 0, 1]])
# sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
# print(sparsity)
# csr_sample = csr_matrix(sample)
# print(csr_sample) # no sparse value, values are assigned as rows and column index

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace = True)

# use KNN to compute similarity with cosine distance metric (over pearson coefficient)
knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 20, n_jobs = -1)
knn.fit(csr_data)

# recommendation algorithm function
# check if movie name input is in database
# if it exists in database, use recommendation system to find similar movies
# then we sort them based on their similarity distance and output top 10
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 5
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list): # if there is a match
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors = n_movies_to_reccomend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key = lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0], 'Distance':val[1]})
        df = pd.DataFrame(recommend_frame, index = range(1, n_movies_to_reccomend + 1))
        return df
    else:
        return "No movies found. Please try another movie!"

print("Movie Recommendation System")
rec = True
while(rec):
    movie_name = input("Enter a movie to get similar recommendations: ")
    print(get_movie_recommendation(movie_name))
    ans = input("Would you like to get more recommendations? ")
    if(ans != "y" and ans != "Y" and ans != "Yes" and ans != "yes" and ans != "YES"):
        rec = False
        print("Thank you, come back soon!")
