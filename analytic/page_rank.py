import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import pandas as pd
import pickle
from tqdm import tqdm

def pageRank(A, n, s = .1, maxerr = .0001, max_iteration=1000):
    """
    Modified page rank to use sparse matrix
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
       Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
       to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    n = A.shape[0]

    # transform G into markov matrix A
    # A = csc_matrix(G,dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    print(rsums.shape)
    # rsums = csc_matrix(rsums,dtype=np.float)
    ri, ci = A.nonzero()
    print(len(ri), len(ci))

    # rsums[rz] = 1.0
    # A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums==0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    count = 0
    loss_history = []
    loss = np.sum(np.abs(r-ro))

    while loss > maxerr:
        ro = r.copy()

        # calculate each pagerank at a time
        for i in tqdm(range(0,n)):
            # inlinks of state i
            Ai = np.array(A[:,i].todense())[:,0]
            # account for sink states
            Di = sink / float(n)
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )

        loss = np.sum(np.abs(r-ro))
        print("Epoch %d , Loss : %f"%(count, loss))
        loss_history.append(loss)
        count += 1
        if count > max_iteration:
            break
        
    # return normalized pagerank
    return r/float(sum(r)), loss_history


def build_author_graph(df_name="author_matrix.hdf5"):
    '''
        Author matrix is a 

    '''
    df = pd.read_hdf(df_name, "all")
    df = df.sort_values(['hIndex'], ascending=False)
    N = len(df)

    author2idx = {}
    idx2author = {}
    author2HIndex = {}
    
    for _, row in df.iterrows():
        idx = len(author2idx)
        authorID = str(row['authorID'])
        author2idx[authorID] = idx
        author2HIndex[authorID] = row['hIndex']
        idx2author[idx] = authorID

    sparse_matrix = {}
    for _, row in df.iterrows():
        authorID = str(row['authorID'])
        weight = row['hIndex']
        startNodeIndex = author2idx[authorID]
        if startNodeIndex not in sparse_matrix:
            sparse_matrix[startNodeIndex] = {}
        for influencedBy in row['influencedIDList']:
            if influencedBy in author2idx:
                idx = author2idx[influencedBy]
                if idx not in sparse_matrix[startNodeIndex]:
                    sparse_matrix[startNodeIndex][idx] = 0
                sparse_matrix[startNodeIndex][idx] += author2HIndex[authorID]

        for influenced in row['influencedByIDList']:
            if influenced in author2idx:
                idx = author2idx[influenced]
                if idx not in sparse_matrix:
                    sparse_matrix[idx] = {}
                if startNodeIndex not in sparse_matrix[idx]:
                    sparse_matrix[idx][startNodeIndex] = 0
                sparse_matrix[idx][startNodeIndex] += author2HIndex[influenced]
    print("Finish reading sparse matrix")
    data = []
    col_series = []
    row_series = []

    for row, row_value in sparse_matrix.items():
        if len(row_value) == 0:
            print("No outlinks at row %d" % row)
        for col, value in row_value.items():
            row_series.append(row)
            col_series.append(col)
            data.append(value)
    data = np.array(data)
    data = (data - data.min())/(data.max() - data.min())
    row_series = np.array(row_series)
    col_series = np.array(col_series)
    print(len(col_series))
    matrix = coo_matrix((data, (row_series, col_series)), shape=(N, N), dtype=np.float).tocsc()

    return matrix, author2idx, idx2author, N

if __name__ == "__main__":
    matrix, author2idx, idx2author, N = build_author_graph()
    print("Start page rank")
    rank, loss_history = pageRank(matrix, N)
    max_ranker = np.argmax(rank)
    print(idx2author[max_ranker])
    pickle.dump({
        'rank': rank,
        'author2idx': author2idx,
        'idx2author': idx2author,
        'loss_hist': loss_history
    }, open("author_rank.pkl", "wb"))

