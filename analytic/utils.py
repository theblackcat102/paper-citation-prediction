import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import merge_dataframe, most_influencial_author, least_influencial_author
import re
import category_encoders as ce
import pickle

def mse(y_pred, y_target):
    y_pred = np.array(y_pred).flatten()
    y_target = np.array(y_target).flatten()
    output_err = np.average((y_target - y_pred)**2, axis=0)
    return output_err

def get_author_page_rank(author_page_rank, author_id):
    rank = len(author_page_rank['author2idx'])
    if author_id  not in author_page_rank['author2idx']:
        return rank+1
    idx = author_page_rank['author2idx'][author_id]
    return rank[idx]

def process_venue(venue):
    venue = venue.replace("year", "")
    venue = venue.replace("J." , "").strip()
    if venue is not None:
        v = venue
        v = re.sub(r'[0-9]+', '', v)
        if v == 'ArXiv':
            v = 'None'
    else:
        v = 'None'
    if len(v) == 0:
        v = 'None'
    v = v.replace("J." , "")
    return v


def load_data(year, filter=None):
    print("load data from dataframe")
    author2rank = pickle.load(open("dictionary/author2rank.pkl", "rb"))
    author_page_rank = pickle.load(open("dictionary/author_rank.pkl", "rb"))

    paper_topic_diversity = pickle.load(open("dictionary/paper_topic_diversity.pkl", 'rb'))
    paper_diversity = pickle.load(open("dictionary/paper_diversity.pkl", 'rb'))

    papers = pd.read_hdf("data/papers.hdf5", str(year))
    authors = pd.read_hdf("data/authors.hdf5", str(year))
    merge1 = merge_dataframe(papers, authors)
    merge2 = most_influencial_author(merge1, authors)
    merge2 = least_influencial_author(merge2, authors)
    venue = []
    main_category = []
    months = []
    days = []
    affiliation = []
    diversity = []
    author_rank = []
    page_rank = []
    venue_topic_diversity = []
    venue_rank = []
    affiliation_encoder = {'unk': -1}
    for idx, row in merge2.iterrows():
        category = row['category']
        if len(category) != 0:
            main_category.append(category[0])
        else:
            main_category.append('None')

        if row['authorID'] in author2rank:
            author_rank.append(author2rank[row['authorID']])
        else:
            author_rank.append(author2rank[-1])
        if len(row['affiliation']) == 0:
            affiliation.append(affiliation_encoder['unk'])
        else:
            if row['affiliation'][0] not in affiliation_encoder:
                idx = len(affiliation_encoder)
                affiliation_encoder[row['affiliation'][0]] = idx
            affiliation.append(affiliation_encoder[row['affiliation'][0]])
        page_rank.append(get_author_page_rank(author_page_rank, row['authorID']))
        date = row['publishedDate']
        months.append(date.month)
        days.append(date.day)
        diversity.append(len(row['topics']))
        v = process_venue(row['venue'])
        venue.append(v)
        venue_rank.append(paper_diversity[v])
        venue_topic_diversity.append(paper_topic_diversity[v])

    merge2['venue_rank'] = venue_rank
    merge2['venue_topic_diversity'] = venue_topic_diversity
    merge2['affiliation'] = affiliation
    merge2['page_rank'] = page_rank
    merge2['main_category'] = main_category
    # merge2.main_category = pd.Categorical(merge2.main_category)
    # merge2['main_category_id'] = merge2.main_category.cat.codes
    merge2['day'] = days
    merge2['authorRank'] = author_rank
    merge2['month'] = months
    merge2['venue'] = venue
    merge2['diversity'] = diversity
    return merge2


