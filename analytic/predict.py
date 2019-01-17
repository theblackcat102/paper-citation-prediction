from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import pickle
import os, glob
from sklearn.externals import joblib 
import category_encoders as ce

from sklearn.decomposition import PCA
from utils import mse, load_data
from ensemble_models import build_model

const_selected_feature = [
    # first author
    'influenceCount', 
    'totalPaper', 'hIndex', 
    # 'authorID',
    'influenceByCount', 'influenceCount_x', 'influenceByCount_y',
    'maxEstCitationAcceleration', 'minEstCitationAcceleration', 'estCitationAcceleration',
    # highest hindex author
    'totalPaper_x', 'hIndex_x', 
    # 'authorID_x',
    'influenceByCount_x',
    # smallest hindex author
    # 'influenceCount_y', 
    'totalPaper_y', 'hIndex_y', 
    # 'authorID_y',
    'influenceByCount_y',
    # other meta info
    'authorCount', 'referencesCount', 
    #'affiliation',
    'day', 'month',
    #'page_rank',
    'diversity',
    'authorRank',
    'figures','table', 'pages', 
    #'main_category_id'
    ]

categorical_feature = ['venue']

selected_feature = const_selected_feature

# def build_model():
#     return LinearRegression()


def train(target_year):
    print(target_year)
    cache_path = './cache'
    if os.path.exists(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year))):
        df = pickle.load(open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year)), 'rb'))
    else:
        df = load_data(target_year)
        pickle.dump( df, open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year)), 'wb'))
    selected_feature = const_selected_feature
    X = df[selected_feature].values
    bde = joblib.load('dictionary/category_encoder.pkl')
    # bde = ce.BinaryEncoder(cols=categorical_feature, return_df=False)
    venue_embedding = bde.transform(df[['venue']]).values
    X = np.hstack((X, venue_embedding))

    nlp_pipeline = joblib.load('dictionary/nlp_pipeline.pkl')
    embeddings = nlp_pipeline.transform(df['summary'].values)
    X = np.hstack((X, embeddings))

    y = df['citationCount'].values
    y = np.clip( y, 0, 41)



    kf = KFold(n_splits=5)
    accuracies = []
    r2_correlations = []
    
    for train_index, test_index in tqdm(kf.split(y)):
        ensemble = build_model()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ensemble.fit(X_train, y_train)
        # y_pred = ensemble.predict(X_test)
        y_pred = [np.mean(y_train)]*len(y_test)
        mse_score = mse(y_pred, y_test)
        r2 = r2_score(y_test, y_pred)
        r2_correlations.append(r2)
        accuracies.append(mse_score)    

    print("MSE: {} ({})".format(np.mean(accuracies), np.std(accuracies)))
    print("R2 : {} ({})".format(np.mean(r2_correlations), np.std(r2_correlations)))
    print("Sample : ")
    print(y_pred[:10])
    print(y_test[:10])

    # if os.path.exists(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year+1))):
    #     test_df = pickle.load(open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year+1)), 'rb'))
    # else:
    #     test_df = load_data(target_year+1)
    #     pickle.dump( test_df, open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year+1)), 'wb'))

    # test_X = test_df[selected_feature].values
    # venue_embedding = bde.transform(test_df[['venue']]).values
    # test_X = np.hstack((test_X, venue_embedding))
    # nlp_pipeline = joblib.load('nlp_pipeline.pkl')
    # embeddings = nlp_pipeline.transform(test_df['summary'].values)
    # test_X = np.hstack((test_X, embeddings))
    # test_y = test_df['citationCount'].values
    # test_y = np.clip( test_y, 0, 25)

    # ensemble = build_model()
    # ensemble.fit(X, y)

    # y_pred = ensemble.predict(test_X)
    # r2 = r2_score(test_y, y_pred)

    # print("R2 : {}".format(r2))
    # joblib.dump(ensemble, "ensemble_model{}.pkl".format(X.shape[1]))

def generate_model(target_year):
    cache_path = './cache'
    if os.path.exists(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year))):
        df = pickle.load(open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year)), 'rb'))
    else:
        df = load_data(target_year)
        pickle.dump( df, open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year)), 'wb'))
    selected_feature = const_selected_feature
    X = df[selected_feature].values
    bde = joblib.load('category_encoder.pkl')
    # bde = ce.BinaryEncoder(cols=categorical_feature, return_df=False)
    venue_embedding = bde.transform(df[['venue']]).values
    X = np.hstack((X, venue_embedding))

    nlp_pipeline = joblib.load('dictionary/nlp_pipeline.pkl')
    embeddings = nlp_pipeline.transform(df['summary'].values)
    X = np.hstack((X, embeddings))
    y = df['citationCount'].values

    y = np.clip( y, 0, 25)
    ensemble = build_model()
    ensemble.fit(X, y)
    joblib.dump(ensemble, "ensemble_{}.pkl".format(target_year))


if __name__ == "__main__":
    # train_category()
    train(2017)
    # generate_model(2017)