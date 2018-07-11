#7/10/18 
#This code classifies an artist's lyrics using TF-IDF and multinomial naive bayes. 
#We can also input a random artist's lyrics into the algorithm and the naive bayes classifier
#will assign it to a similar artist 

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import pandas as pd 
import sklearn 
from sklearn.feature_selection import chi2

#import dataset 
lyrics=pd.read_csv("lyrics.csv",encoding="latin-1")
lyrics1=lyrics.dropna()
lyrics1['genre'].unique() 

rap_lyrics=lyrics1[lyrics1['genre']=='Hip-Hop']
rap_lyrics.shape #(24850,6) 

#assign unique id to each artist 
rap_lyrics['artist_id']=rap_lyrics.groupby(['artist']).ngroup()
rap_lyrics['artist_id'].unique() 

#tfidf vectorizer 
tfidf=TfidfVectorizer(sublinear_tf=True,norm='l2',
encoding='latin-1',ngram_range=(1,2),stop_words='english')

features=tfidf.fit_transform(rap_lyrics.lyrics).toarray() 
labels=rap_lyrics.artist_id 
features.shape #(24850,2560515) (bi-grams and tri-grams)

#multi-class classifier libraries 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#train and test sets for classification 
X_train,X_test,y_train,y_test=train_test_split(rap_lyrics['lyrics'],
rap_lyrics['artist_id'],random_state=0)

#count vector 
count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(X_train) #18637 x 138236 sparse matrix 
#tfidf 
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts) #tfidf-score matrix 
X_train_tfidf 

#multinomial naive bayes classifier 
clf_nb=MultinomialNB().fit(X_train_tfidf,y_train)
#input test lyrics and return the most similar artist
clf_nb.predict(count_vect.transform([song2]))
#search artist 
rap_lyrics[rap_lyrics.artist_id==881]['artist'] #returns the artist name

# cluster artists
rap_lyrics_sub=rap_lyrics[['lyrics','artist']]
rap_lyrics_sub['artist_id']=rap_lyrics_sub.groupby(['artist']).ngroup()

rap_lyrics_song=rap_lyrics_sub['lyrics']
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix=tfidf_vectorizer.fit_transform(rap_lyrics_song)
terms=tfidf_vectorizer.get_feature_names() 

#K-means clustering 
from sklearn.cluster import KMeans
num_clusters=5 #five cluster groups 
k_means=KMeans(n_clusters=num_clusters)
k_means.fit(tfidf_matrix)
clusters=k_means.labels_.tolist() 

cluster_results=pd.DataFrame(clusters) 
cluster_results.columns=['cluster']

final_cluster=pd.concat([cluster_results,rap_lyrics_sub],axis=1)
final_cluster.head(10)
final_cluster.info() 
final_cluster['cluster'].value_counts() 

#cluster metrics 
#cluster one 
cluster_one=final_cluster[final_cluster.cluster==0]
cluster_one['artist'].value_counts() 
cluster_two=final_cluster[final_cluster.cluster==1]





