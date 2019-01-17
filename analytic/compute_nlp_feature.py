import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import random 
from utils import load_data
random.seed(13)

#visualization packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
import os, glob
from sklearn.externals import joblib 

n_features = 1000
n_topics = 8
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def tfidf_nmf(year=2018, n_features = 20000, n_top_words = 10, n_topics=16):
    df = pd.read_hdf('papers.hdf5', '2018')
    for year in range(1990, 2018):
        tf = pd.read_hdf('papers.hdf5', str(year))
        df.append(tf)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,max_features=n_features,stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(df['summary'])
    joblib.dump(tfidf_vectorizer, 'tfidf.pkl')

    nmf = NMF(n_components=n_topics, random_state=0,alpha=.1, l1_ratio=.1).fit(tfidf)

    print("Topics found via NMF:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)
    nmf_embedding = nmf.transform(tfidf)
    joblib.dump(nmf, 'nmf.pkl')
    nmf_embedding = (nmf_embedding - nmf_embedding.mean(axis=0))/nmf_embedding.std(axis=0)

    nlp_pipeline = Pipeline(memory=None,
         steps=[('tfidf', tfidf_vectorizer),
                ('nmf', nmf)])
    joblib.dump(nlp_pipeline, 'nlp_pipeline.pkl')
    top_idx = np.argsort(nmf_embedding,axis=0)[-3:]

    count = 0
    for idxs in top_idx.T: 
        print("\nTopic {}:".format(count))
        for idx in idxs:
            print(df.iloc[idx]['title'])
        count += 1
        if count > 10:
            break



if __name__ == "__main__":
    tfidf_nmf()