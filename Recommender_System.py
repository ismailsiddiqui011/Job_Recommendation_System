#!/usr/bin/env python
# coding: utf-8

# # Job Recommendation System

# ## Importing Libraries

# In[1]:
def predict(user):
        
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity
        import operator
        
        
        # ## Loading Data
        
        # In[2]:
        
        
        jobs = pd.read_csv('jobs.csv')
        
        
        # In[3]:
        
        
        jobs.isnull().sum() # Checking For Null values
        
        
        # In[4]:
        
        
        job_click = pd.read_csv('job_clicks.csv')
        
        
        # In[5]:
        
        
        job_click.isnull().sum() # Checking For Null values
        
        
        # ## Preparing Data For Recommendation System
        
        # In[6]:
        
        
        # avg number of job rated per user
        click_per_user = job_click.groupby('userId')['Clicks'].count()
        
        
        # In[7]:
        
        
        # avg number of ratings given per job
        click_per_job = job_click.groupby('jobId')['Clicks'].count()
        
        
        # In[8]:
        
        
        # counts of ratings per job as a df
        click_per_job_df = pd.DataFrame(click_per_job)
        
        filtered_click_per_job_df = click_per_job_df[click_per_job_df.Clicks >= job_click.Clicks.mean()] # Using Mean Clicks per job
        # build a list of job_ids to keep
        popular_job = filtered_click_per_job_df.index.tolist()
        
        
        # In[9]:
        
        
        # counts ratings per user as a df
        click_per_user_df = pd.DataFrame(click_per_user)
        
        filtered_click_per_user_df = click_per_user_df[click_per_user_df.Clicks >= 1] # Using 1 means treating all users same and can be modified for more prolific users
        # build a list of user_ids to keep
        prolific_users = filtered_click_per_user_df.index.tolist()
        
        
        # In[10]:
        
        
        filtered_ratings = job_click[job_click.jobId.isin(popular_job)] # Filtering
        filtered_ratings = job_click[job_click.userId.isin(prolific_users)]
        len(filtered_ratings)
        
        
        # In[11]:
        
        
        rating_matrix = filtered_ratings.pivot_table(index='userId', columns='jobId', values='Clicks') # Making Dataframe of filtered data
        # replace NaN values with 0
        rating_matrix = rating_matrix.fillna(0)
        # display the top few rows
        rating_matrix.head(2)
        
        
        # ## Building Recommendation Sysytem
        
        # In[12]:
        
        
        def similar_users(user_id, matrix, k=3):
            # create a df of just the current user
            user = matrix[matrix.index == user_id]
            
            # and a df of all other users
            other_users = matrix[matrix.index != user_id]
            
            # calc cosine similarity between user and each other user
            similarities = cosine_similarity(user,other_users)[0].tolist()
            
            # create list of indices of these users
            indices = other_users.index.tolist()
            
            # create key/values pairs of user index and their similarity
            index_similarity = dict(zip(indices, similarities))
            
            # sort by similarity
            index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
            index_similarity_sorted.reverse()
            
            # grab k users off the top
            top_users_similarities = index_similarity_sorted[:k]
            users = [u[0] for u in top_users_similarities]
            
            return users
        
        
        # In[13]:
        
        
        similar_user_indices = similar_users(user, rating_matrix)
        
        
        # In[14]:
        
        
        def recommend_jobs(user_index, similar_user_indices, matrix, items=5):
            
            # load vectors for similar users
            similar_users = matrix[matrix.index.isin(similar_user_indices)]
            # calc avg ratings across the 3 similar users
            similar_users = similar_users.mean(axis=0)
            # convert to dataframe so its easy to sort and filter
            similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
            
            
            # load vector for the current user
            user_df = matrix[matrix.index == user_index]
            # transpose it so its easier to filter
            user_df_transposed = user_df.transpose()
            # rename the column as 'rating'
            user_df_transposed.columns = ['rating']
            # remove any rows without a 0 value. job not seen yet
            user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
            # generate a list of jobs the user has not seen
            new_job = user_df_transposed.index.tolist()
            
            # filter avg ratings of similar users for only job the current user has not seen
            similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(new_job)]
            # order the dataframe
            similar_users_df_ordered = similar_users_df.sort_values(by=['mean'], ascending=False)
            # grab the top n job   
            top_job = similar_users_df_ordered.head(items)
            top_job_indices = top_job.index.tolist()
            # lookup these jobs in the other dataframe to find names
            job_info = jobs[jobs['jobID'].isin(top_job_indices)]
            
            
            return job_info #items
        
        return recommend_jobs(user, similar_user_indices, rating_matrix)
        # In[15]:
        