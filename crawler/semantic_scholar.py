import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

SEMANTIC_URLS = 'https://api.semanticscholar.org/v1/paper/arXiv:'
SEMANTIC_AUTHOR1 = 'https://www.semanticscholar.org/api/1/author/'
SEMANTIC_AUTHOR2 = 'https://api.semanticscholar.org/v1/author/'


session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

def get_arvixpaper_semantic_scholar(arvixId):
    URL = SEMANTIC_URLS + arvixId
    res = session.get(URL)
    if res.status_code >= 300:
        return False, {}
    results = res.json()
    return True, results

def get_author_data(authorID):
    # get the official json data first
    authorID = str(authorID)
    URL = SEMANTIC_AUTHOR2 + authorID
    res = session.get(URL)
    if res.status_code >= 300:
        return False, {}
    results1 = res.json()

    URL = SEMANTIC_AUTHOR1 + authorID
    res = session.get(URL)
    if res.status_code >= 300:
        return False, {}
    results2 = res.json()

    if 'author' in results2:
        authorData = results2['author']
        results = {**authorData, **results1}
        return True, results
    return False, {}

if __name__ == "__main__":
    import json
    success, result = get_author_data(2569825)
    with open("author.txt", "w") as f:
        f.write(json.dumps(result))

    # success, result = get_arvixpaper_semantic_scholar('1811.04091')
    # if success is False:
    #     print("Paper failed")
    # else:
    #     print(result['authors'][0])
    #     with open("paper.txt", "w") as f:
    #         f.write(json.dumps(result))