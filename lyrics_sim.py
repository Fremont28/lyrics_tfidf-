#cosine similarity lyrics
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
lyrics=pd.read_csv("lyrics.csv",encoding="latin-1")
lyrics1=lyrics.dropna()

#scale-down dataset 
lyrics1=lyrics[0:700000]
lyrics2=lyrics1.dropna() 

lyrics_sub=lyrics2[["artist","song","year","genre","lyrics"]][0:50000]
lyrics_sub.shape 

#unique artists 
artists=lyrics_sub.artist.value_counts() #2843 artists 
genre=lyrics_sub.genre.value_counts() #type of music genres 

#split the data in train and test 
lyrics3=lyrics2['lyrics'][0:50000]

lyrics4=lyrics4[0:5000] #sample 5000 artists 
tfidf_vectorizer=TfidfVectorizer() 
tfidf_matrix=tfidf_vectorizer.fit_transform(lyrics4)
tfidf_matrix.shape #(5000,40019) #5000 songs (rows of the matrix), 40019 tf-idf terms (columns of the matrix)
cos_sim=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)

#most similar artists to Beyonce 
np.where(cos_sim>0.10)

