# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:34:17 2023

@author: weihau.yap
"""

from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import requests
import re
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

#Define sub functions
def Data_Fetching(file_path):
    data = pd.read_csv(file_path)
    #data = data.dropna()
    #st.write(data)
    #st.write("Rows, Columns: ", data.shape)
    return data
        
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#Set up front end
#Webpage config
st.set_page_config(page_title = "Movie With Me!", page_icon=":movie_camera:", layout = "wide")
with st.container():
    
    left_column, right_column  = st.columns(2)
    with left_column:
        st.title("Movie Recommender with Ranking System")

    with right_column:
        lottie_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_khzniaya.json")
        st_lottie(lottie_animation, key = "Animation", speed = 1.3, width = 200)


st.write("---")

st.subheader("Highlights of this project:")
st.write("""
         1. Interactive search engine to simulate ONE's favourite movie list.
         2. Collaborative Filtering based recommendation system.
         3. Ranking feature in terms of similarity score to further enhance the recommendation system.
         4. Generate a tailored Movie Recommendation List for the users based on their favourite movie list!
         """)
st.write("---")

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

file_path = current_dir / "ratings_lite.csv"
ratings = Data_Fetching(file_path)
#st.write(ratings)

file_path1 = current_dir / "movies.csv"
movie = Data_Fetching(file_path1)
#st.write(movie)

#Make sure movie id is string for primary key to join df
ratings['movieId'] = ratings['movieId'].astype(str)
movie['movieId'] = movie['movieId'].astype(str)
movie["clean_title"] = movie["title"].apply(clean_title)

#Generate a list of watched movie
with st.form("Data Entry", clear_on_submit=True):
    st.number_input(label='Recommendation Based On Your Top Movie List --> Top 10, Top 50, Top 100 etc.', value=int(), key="number", step=10)
    
    "---"
    submitted = st.form_submit_button("Generate!")
if submitted:
    number = st.session_state["number"]
    
    mymovie = movie.sample(n = number)
    movie_set = set(mymovie['movieId'])
    mymovie_ui = mymovie.drop(['movieId','clean_title'], axis = 1)
    mymovie_ui.rename(columns={'title': 'Title', 'genres': 'Genres'}, inplace = True)
    mymovie_ui.sort_index(inplace = True)
    st.subheader("Generated User's Favourite Movie List. :love_letter:")
    
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """
    
    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
    
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    #st.write(mymovie_ui)
    st.dataframe(mymovie_ui)
    st.write("Rows, Columns: ", mymovie_ui.shape) 
    
    #Find similar users
    overlap_users = {}
    for i in range(ratings.shape[0]):
        if ratings['movieId'][i] in movie_set:
            if ratings['userId'][i] not in overlap_users:
                overlap_users[ratings['userId'][i]] = 1
            else:
                overlap_users[ratings['userId'][i]] += 1
    
    #Choose only 10% of most similar users to us
    filtered_overlap_users = set([k for k in overlap_users if overlap_users[k] > len(movie_set)*0.1])
    
    #Get the movie id and ratings from the overlap users
    interactions = ratings[(ratings["userId"].isin(filtered_overlap_users))][["userId", "movieId", "rating"]]
    
    #Get my movie ratings
    my_movie_ratings = ratings[(ratings["movieId"].isin(movie_set))][["movieId", "rating"]]
    myuserId = pd.Series(-1, index = range(len(my_movie_ratings)))
    my_movie_ratings = my_movie_ratings.merge(myuserId.rename("userId"), left_index = True, right_index = True)
    my_movie_ratings = my_movie_ratings.reindex(columns = ['userId', 'movieId', 'rating'])
    
    #Adding own movie ratings into the ratings list
    interactions = pd.concat([my_movie_ratings[["userId", "movieId", "rating"]], interactions])
    interactions['userId'] = interactions['userId'].astype(str)
    interactions['movieId'] = interactions['movieId'].astype(str)
    interactions['rating'] = pd.to_numeric(interactions['rating'])
    interactions['user_index'] = interactions['userId'].astype('category').cat.codes
    interactions['movie_index'] = interactions['movieId'].astype('category').cat.codes
    
    #Store in sparse matrix for memory usage efficiency
    ratings_mat_coo = coo_matrix((interactions['rating'], (interactions['user_index'], interactions['movie_index'])))
    ratings_mat = ratings_mat_coo.tocsr()
    
    #interactions[interactions['userId'] == '-1']
    my_index = 0
    
    #Calculate how similar the other users to us
    similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()
    
    #Get user id for the users most similar to us
    if (len(similarity) - 1) % 2 == 0:
        number = len(similarity - 1)
    else:
        number = len(similarity)
    
    #5 users that are most similar to us
    indices = np.argpartition(similarity, -5)[5:]
    
    similar_users = interactions[interactions['user_index'].isin(indices)].copy()
    similar_users = similar_users[similar_users['userId'] != '-1']
    
    #Find how many times each movies appear in this recommendation
    movie_recs = similar_users.groupby('movieId').rating.agg(['count' , 'mean'])
    
    movie_recs = movie_recs.merge(movie, how = 'inner', on = 'movieId')
    
    #Find the number of times a movie is rated
    total_ratings_per_movie = ratings.groupby('movieId').rating.agg(['count'])
    
    movie_recs = movie_recs.merge(total_ratings_per_movie, how = 'inner', on = 'movieId')
    movie_recs.rename(columns={'count_x': 'count', 'count_y': 'rating'}, inplace = True)
    
    #Ranking recommendations
    movie_recs['adjusted_count'] = movie_recs['count'] * (movie_recs['count'] / movie_recs['rating'])
    movie_recs['score'] = movie_recs['mean'] * movie_recs['adjusted_count']
    #Take out any movie we have already watched
    movie_recs = movie_recs[~movie_recs['movieId'].isin(mymovie['movieId'])]
    #Take out any movie title too because this is not the cleanest data, different movies might have the same movie id
    movie_recs = movie_recs[~movie_recs['clean_title'].isin(mymovie['clean_title'])]
    #At least n users who are similar to us had watch the movie and like it in order to put into the movie reco list
    movie_recs = movie_recs[movie_recs['count'] > 1]
    
    mean = (movie_recs['mean'].max() + movie_recs['mean'].min()) / 2
    movie_recs = movie_recs[movie_recs['mean'] >= mean]
    
    score = (movie_recs['score'].max() + movie_recs['score'].min()) / 2
    movie_recs = movie_recs[movie_recs['score'] >= score]
    
    top_movie_recs = movie_recs.sort_values('score', ascending = False)
    top_movie_recs_ui = top_movie_recs.drop(['movieId','clean_title','count','mean', 'rating', 'adjusted_count'], axis = 1)
    top_movie_recs_ui.rename(columns={'title': 'Title', 'genres': 'Genres', 'score': 'Score'}, inplace = True)
    
    st.subheader("Here's the movie recommendation list for you! :partying_face:")
    #st.write(top_movie_recs_ui)
    st.dataframe(top_movie_recs_ui)
    st.write("Rows, Columns: ", top_movie_recs_ui.shape)
    
    with st.form("Re-Run", clear_on_submit=True):
        rerun = st.form_submit_button("Rerun?")
    if rerun:
        st.experimental_rerun()   
    