#!/usr/bin/env python
# coding: utf-8

# ## User Based Recommendation 
# ## Data Preparation

# In[156]:


## Step 1: Read movie and rating data sets.
## The dataset is provided by MovieLens, a movie recommendation service. It contains movies along with their ratings.
import pandas as pd

movie = pd.read_csv('movie.csv',nrows=300000)
rating = pd.read_csv('rating.csv',nrows=300000)
## merge two data sets
df = movie.merge(rating, how="left", on="movieId")
df.head()


# In[157]:


comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()


# In[158]:


rare_movies = comment_counts[comment_counts["count"] < 10000]
rare_movies.index


# In[159]:


common_movies = df[~df["title"].isin(rare_movies)]
common_movies.head()


# In[160]:


##Step 4: Create a pivot table where the index consists of userIDs,
##the columns consist of movie titles, and the values are the ratings.
user_movie_df = common_movies.pivot_table(values="rating", index="userId", columns="title")
user_movie_df


# In[165]:


## Step 5: Functionalize all operations performed.
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("movie.csv", nrows=300000)
    rating = pd.read_csv("rating.csv",nrows=300000 )
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] < 10000]
    rare_movies.index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(values="rating", index="userId", columns="title")
    return user_movie_df
create_user_movie_df()


# ## Determining the Movies Watched by the User to Be Suggested

# In[174]:


## Choose a random user id.
random_user = 500
## Create a new dataframe named random_user_df consisting of observation units belonging to the selected user.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df 


# In[175]:


# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = list(random_user_df.columns[random_user_df.notna().any()])
movies_watched


# In[176]:


## Accessing Data and IDs of Other Users Watching the Same Movies

movies_watched = list(random_user_df.columns[random_user_df.notna().any()])
len(movies_watched)


# ## Accessing Data and IDs of Other Users Watching the Same Movies

# In[177]:


## Select the columns of movies watched by the selected user from user_movie_df and create a new dataframe named 
## movies_watched_df.

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df


# In[178]:


## Create a new dataframe named user_movie_count that contains information about how many of the movies
## the selected user has watched for each user.
user_movie_count = movies_watched_df.T.notnull().sum() #Bu bir Series
user_movie_count = user_movie_count.reset_index() #Bu bir DataFrame
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count


# In[179]:


user_movie_count[user_movie_count["userId"] == random_user]
user_movie_count


# In[180]:


## Create a list named users_same_movies from the user IDs of those who watched 
## 60 percent or more of the movies voted by the selected user.
percentage = 60
user_same_movies = user_movie_count[user_movie_count["movie_count"] > (len(movies_watched) * percentage / 100)]["userId"]
user_same_movies


# ## Determining the Most Similar Users to the User to be Recommended

# In[181]:


## Filter the movies_watched_df dataframe to find the IDs of users that are
## similar to the selected user in the user_same_movies list.
final_df = movies_watched_df[movies_watched_df.index.isin(user_same_movies)]

final_df[final_df.index == random_user]
final_df


# In[195]:


corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userId_1", "userId_2"]
corr_df = corr_df.reset_index()
corr_df[corr_df["userId_1"] == random_user]


# In[198]:


corr_th = 0.30
top_users = corr_df[(corr_df["userId_1"] == random_user) & (corr_df["corr"] > corr_th)][["userId_2", "corr"]].sort_values("corr", ascending=False)
top_users.columns = ["userId", "corr"]
top_users_ratings = top_users.merge(rating, how="left", on="userId")
top_users_ratings 


# ## Weighted Average Recommendation Score

# In[200]:


##  corr x rating = weighted_rating
top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]
top_users_ratings


# In[202]:


## A dataframe named recommendation_df 
## containing the movie id and the average value of the weighted ratings of all users for each movie.
recommendation_df = top_users_ratings[["movieId", "weighted_rating"]]
recommendation_df


# In[207]:


## Select the movies with a weighted rating greater than 1.5
## in recommendation_df and sort them according to their weighted rating.
rating_th = 1.5
movies_to_be_recommended = recommendation_df[recommendation_df["weighted_rating"] > rating_th]. \
    sort_values("weighted_rating", ascending=False).head()

movies_to_be_recommended


# In[208]:


movies_to_be_recommended.merge(movie[["movieId", "title"]])


# ##  Item-Based Recommendation

# In[209]:


## Read movie, rating data sets.
import pandas as pd

movie = pd.read_csv('movie.csv',nrows=300000)
rating = pd.read_csv('rating.csv',nrows=300000)
## merge two data sets
df = movie.merge(rating, how="left", on="movieId")
df.head()


# In[212]:


## Get the id of the movie with the most up-to-date score among the movies that the selected user gave 5 points to.
user = 500
top_rated_movies = df[(df["userId"] == user) & (df["rating"] == 5)]
last_top_rated_movie_id = top_rated_movies.sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
last_top_rated_movie_id


# In[216]:


movie[movie["movieId"] == last_top_rated_movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == last_top_rated_movie_id]["title"].values[0]]


# In[217]:


## Using the filtered dataframe, find the correlation between the selected movie and other movies and rank them.
recommended_movies = user_movie_df.corrwith(movie_df).sort_values(ascending=False)


# In[219]:


recommended_movies[1:6]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




