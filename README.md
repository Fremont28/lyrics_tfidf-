# lyrics_tfidf-
Cosine similarity between song lyrics 

TF-IDF For Mapping Artists’ Lyrics 

We are interested in examining the relationships between different artist’s lyrics using text analysis or more specifically TF-IDF, which stands for term frequency-inverse document frequency. 

The goal of TF-IDF is to reduce the weightage of more common words across text documents. TF-IDF is broken into two components where term frequency is defined as:

Number of times term t appears in a document /total terms in the document  

and inverse term frequency is defined as:

log (total number of documents/number of documents with term t in it) 

The first step in our analysis is to transform each song’s lyrics into TF-IDF form using the TF-IDF vectorization function in Python. The TF-IDF vector has 5,000 rows representing each unique song and 40,019 columns corresponding to unique words (or lyrics), which contain their TF-IDF score.  

The next step is use cosine similarity to compare individual song lyrics. For instance, we could measure how similar Beyoncé’s Beautiful Liar (2007) is to the other 4,999 songs in the dataset. As a reference, cosine similarity scores are on a zero to one scale. 

To save time and crashes, we analyzed just the first 5,000 songs in the dataset (362,575 songs in total). Our reduced dataset features 2,843 unique artists. The most popular genres in the sample are rock (39.3%), pop (14.8%), and hip-hop (9.5%). 

As a test, we measured the cosine similarity between Beyoncé’s Ego Remix (2009) and the other 4,999 songs in the sample. A little unsurprisingly, the most similar lyric matches to Ego Remix are other Beyoncé songs including Than Tell Me (2009), Honesty (2009), and Once in A Lifetime (2009). 

In a future analysis, we will use multi-class classification and K-means clustering to predict and group similar songs and artists. 

