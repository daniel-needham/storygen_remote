import pandas as pd
import os
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def load_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

documents = []
metadata = pd.read_csv('data/ghost_stories/top10_ghoststories.csv')

for index, row in metadata.iterrows():
    #copy file to new location
    file_name = row['id'] + '_text.txt'
    file_path = os.path.join('data/ghost_stories/top10', file_name)
    book = load_text_file(file_path)
    documents.append(book)

df = pd.DataFrame(documents, columns=['text'])

## Preprocessing
stop_words_l = stopwords.words('english')
df['text_clean']=df.text.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]','',w).lower() for w in x.split()))

tfidfvectoriser=TfidfVectorizer()
tfidfvectoriser.fit(df.text_clean)
tfidf_vectors=tfidfvectoriser.transform(df.text_clean)

pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T).toarray()
pairwise_differences=euclidean_distances(tfidf_vectors)

def most_similar(doc_id,similarity_matrix,matrix):
    print (f'Document: {doc_id}')
    print ('\n')
    print ('Similar Documents:')
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        print('\n')
        print (f'Document: {ix}')
        print (f'{matrix} : {similarity_matrix[doc_id][ix]}')

most_similar(0,pairwise_similarities,'Cosine Similarity')
most_similar(0,pairwise_differences,'Euclidean Distance')