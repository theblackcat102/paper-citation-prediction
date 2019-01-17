import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import pickle
from sklearn.externals import joblib 
import category_encoders as ce

import os, glob
from sklearn.decomposition import PCA
from ml_utils import mse, load_data

const_selected_feature = [
    # first author
    'influenceCount', 
    'totalPaper', 'hIndex', 
    # 'authorID',
    'venue_topic_diversity', 'venue_rank',
    'influenceByCount', 'influenceCount_x', 'influenceByCount_y',
    #'maxEstCitationAcceleration', 'minEstCitationAcceleration', 'estCitationAcceleration',
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
    # 'page_rank',
    'diversity',
    'authorRank',
    'figures','table', 'pages', 
    #'main_category_id'
    ]

selected_feature = const_selected_feature
cache_path = './cache'
if os.path.exists(cache_path) is False:
    os.makedirs(cache_path)
target_year = 2014

def get_sentence_embed(titles):
    global selected_feature
    from nlp import sent2vec
    sentences = []
    for t in titles:
        sentences.append(t)
    results  = [ ]
    for idx in range(0, len(sentences), 64):
        embeddings = sent2vec(sentences[idx:idx+64])
        results.append(embeddings)

    embeddings = np.concatenate(results)

    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)
    embedding_size = embeddings.shape[1]
    if 'embeddings_0' not in selected_feature:
        for i in range(embedding_size):
            selected_feature.append('embedding_{}'.format(i))
    return embeddings


def train(model_class, domain=None):
    cache_path = './cache'
    if os.path.exists(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year))):
        df = pickle.load(open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year)), 'rb'))
    else:
        df = load_data(target_year)
        pickle.dump( df, open(os.path.join(cache_path, "{}_boosted_tree.pkl".format(target_year)), 'wb'))
    if domain is not None:
        print("Find important feature in {}".format(domain))
        domain2category = pickle.load(open("domain2category.pkl", 'rb'))
        df = df.loc[df['main_category'].isin(domain2category[domain])]
    selected_feature = const_selected_feature
    X = df[selected_feature].values
    bde = joblib.load('category_encoder.pkl')
    # bde = ce.BinaryEncoder(cols=categorical_feature, return_df=False)
    venue_embedding = bde.transform(df[['venue']]).values
    # pca = PCA(n_components=32)
    # venue_embedding = pca.fit_transform(venue_embedding)
    X = np.hstack((X, venue_embedding))
    venue_embedding_size = venue_embedding.shape[1]
    print(venue_embedding_size)
    for i in range(venue_embedding_size):
        selected_feature.append('venue_'+str(i))

    nlp_pipeline = joblib.load('nlp_pipeline.pkl')
    embeddings = nlp_pipeline.transform(df['summary'].values)
    embeddings_size = embeddings.shape[1]
    print(embeddings_size)
    for i in range(embeddings_size):
        selected_feature.append('title_'+str(i))
    X = np.hstack((X, embeddings))
    y = df['citationCount'].values
    y = np.clip( y, 0, 23)
    kf = KFold(n_splits=5)
    accuracies = []
    print(model_class)
    importance_stats = {}
    for train_index, test_index in tqdm(kf.split(y)):
        model = model_class(verbose=0)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_score = mse(y_pred, y_test)
        accuracies.append(mse_score)
        feature_importance = model.get_feature_importance(prettified=True)
        for feature_id, score in feature_importance:
            if selected_feature[int(feature_id)] not in importance_stats:
                importance_stats[selected_feature[int(feature_id)]] = score
            else:
                importance_stats[selected_feature[int(feature_id)]] += score
    limit = 0
    for key, value in importance_stats.items():
        print(key, " score: ", value)
        limit += 1
        if limit >= 30:
            break
    print("MSE: {} ({})".format(np.mean(accuracies), np.std(accuracies)))
    print("Sample : ")
    print(y_pred[:10])
    print(y_test[:10])


if __name__ == "__main__":
    '''
        2017 
        XGBClassifier MSE: 617.0772263679462 (249.32754058386428)
        XGBRegressor MSE: 499.23793532658954 (213.66789684599874)
        CatBoostRegressor MSE: 490.75306147968405 (212.25437147951337)
        CatBoostClassifier MSE: 626.0802546908217 (253.11501164465525)
        Clip value:
        XGBClassifier MSE: 75.59319339373974 (7.634979885758403)
        XGBRegressor MSE: 41.019533131237374 (4.89291832026593)
        CatBoostRegressor MSE: 40.35242530613481 (4.587054535753539)
        CatBoostClassifier MSE: 69.14407071393363 (10.368940088661704)
        Added feature: authorCount, figure, table, 
        XGBClassifier MSE: 68.09434744338566 (6.128824538465486)
        XGBRegressor MSE: 35.810905593590256 (4.302617219704865)
        CatBoostRegressor MSE: 33.98436223427542 (3.75341603341854)
        CatBoostClassifier MSE: 69.17107336504003 (10.377030964171336)

        2018
        XGBClassifier MSE: 30.652890252907874 (4.787327633671798)
        XGBRegressor MSE: 27.045078591629512 (4.985888298305488)
        CatBoostRegressor MSE: 28.505737953619388 (5.865317777577888)
        CatBoostClassifier MSE: 29.37475483154456 (4.691823301335586)
        
        Added feature
        XGBClassifier MSE: 13.500829160701532 (2.308399110877171)
        XGBRegressor MSE: 9.4800557173621 (1.3760326118363793)
        CatBoostRegressor MSE: 9.197358639798752 (1.3386542170870745)
        CatBoostClassifier MSE: 12.00784514763464 (1.9332583422813232)
    '''
    target_year = 2017
    print(target_year)
    model_list = [CatBoostRegressor]
    # model_list = [XGBClassifier, XGBRegressor]
    # for cat in ['physics', 'math', 'econ', 'q-bio']:
    for model_class in model_list:
        train(model_class, 'physics')

