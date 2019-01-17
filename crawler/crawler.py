from tqdm import tqdm
import arxivpy
import arxiv
from semantic_scholar import get_arvixpaper_semantic_scholar, get_author_data
from models import Author, Paper
import logging
import time
import os, glob
from xml.etree import ElementTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


category_list = ['cs.AI', 'cs.CL', 'cs.CC', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.GT', 'cs.CV', 'cs.CY',
    'cs.CR', 'cs.DS', 'cs.DB', 'cs.DC', 'cs.ET','cs.HC', 'cs.IR', 'cs.LG', 'cs.IT', 
    'cs.NE', 'cs.RO', 'cs.SD','cs.SI',
    'eess.IV', 'eess.AS', 'eess.SP', 'stat.ML'] + ['cs.MM','cs.NI', 'cs.OH','cs.MS', 'cs.SE','math.NT','math.IT', 'math.OC','math.MS', 'math.ST', 'math.SP', 'math.SG', 'math.RA', 'math.RT', 'math.PR', 'cs.LO','cs.MS', 'cs.MA','cs.OS']
other_category_list = []
start_index = 0
end_index = 3000


def parse_xml_file(filename):
    e = ElementTree.parse(filename).getroot()
    article = {}
    for child in e:
        key = child.tag.split("}")[-1]
        if key in article:
            key = key + '1'
        article[key] = child.text
    article['id'] = article['identifier'].split("/")[-1]
    article['main_author'] = article['creator'].split(",")[0]
    article['abstract'] = article['description']
    if 'description1' in article:
        article['comment'] = article['description1']
    else:
        article['comment'] = 'None'
    if 'withdraw' in article['comment'].lower():
        return None
    if 'withdraw' in article['description'].lower():
        return None
    article['authors'] = article['creator']
    article['publish_date'] = article['date']
    article['journal_ref'] = ''
    article['url'] = article['identifier']
    article['term'] = 'None'

    return article

def update_paper():
    idx = 0
    for filename in tqdm(glob.glob("oai/*.xml")):
        article = parse_xml_file(filename)
        if article is None or idx < 346728:
            idx += 1
            continue
        arvixID = article['id'].split('v')[0]
        query = Paper.select().where(Paper.arvixID == arvixID)
        if query.exists():
            continue
        success, article_meta = get_arvixpaper_semantic_scholar(arvixID)
        if success is False:
            logging.debug("Paper not exists in semantic scholar, arvixID : %s" % arvixID)
            continue
        authorIDList = [ int(author['authorId']) if author['authorId'] is not None else -1 for author in article_meta['authors']  ]
        authorNames = [article['main_author']]
        authorCount = len(article_meta['authors'])
        if authorCount > 1:
            other_author = [ name.strip() for name in article['authors'].split(',') if len(name) > 1 and name != article['main_author']]
            authorNames += other_author
        paper_category = [article['term']]
        try:
            paper = Paper.create(
                indexID=idx,
                arvixID = arvixID,
                paperId = article_meta['paperId'],
                doiID = str(article_meta['doi']),

                title= article['title'],
                summary=article['abstract'],
                category=paper_category,
                comments=article['comment'],
                journal_ref=article['journal_ref'],

                url=article['url'],
                authorID=authorIDList,
                authorName=authorNames,
                authorCount=authorCount,
                publishedDate=article['publish_date'],
                citationVelocity=article_meta['citationVelocity'],
                referencesCount=len(article_meta['references']),
                topics=article_meta['topics'],
                venue=str(article_meta['venue']),
                year=article_meta['year'],
                influentialCitationCount=article_meta['influentialCitationCount'],
                citationCount=len(article_meta['citations']),
                citations=article_meta['citations'],
                )
            try:
                for meta in ['page', 'figure', 'table']:
                    if meta in article['comment']:
                        comment = article['comment'].replace(';', ',')
                        for segment in comment.split(','):
                            if meta in segment:
                                page_prefix = segment.split(meta)[0]
                                if meta == 'page':
                                    paper.pages = int(page_prefix.strip())
                                elif meta == 'figure':
                                    paper.figures = int(page_prefix.strip())
                                elif meta == 'table':
                                    paper.table = int(page_prefix.strip())
                                break
            except:
                logging.debug("Error in parsing meta data")
            paper.save()
        except BaseException as e:
            logging.warning("Error in arvix id %s, error: %s" % (arvixID, str(e)))
        time.sleep(0.2)
        idx += 1

def crawl_category(term='cs.LG'):
    index_iteration = 500
    logging.info("Crawling category : %s", term)
    for index in range(start_index, end_index, index_iteration):
        logging.info("\nBatch : %d-%d" % (index, index+index_iteration))
        articles = arxivpy.query(search_query=[term],
                            start_index=index, max_index=index+index_iteration, results_per_iteration=index_iteration,
                            wait_time=0.2, sort_by='lastUpdatedDate')
        article_batch_count = len(articles)
        if article_batch_count == 0:
            logging.warning('Article not found in batch %d - %d' % (index, index+index_iteration))
        for idx, article in tqdm(enumerate(articles), total=article_batch_count):
            arvixID = article['id'].split('v')[0]
            query = Paper.select().where(Paper.arvixID == arvixID)
            if query.exists():
                paper = Paper.get(Paper.arvixID==arvixID)
                categories = paper.category
                if term not in categories:
                    categories.append(term)
                Paper.update(category=categories).where(Paper.arvixID == arvixID).execute()
                continue
            success, article_meta = get_arvixpaper_semantic_scholar(arvixID)
            if success is False:
                logging.debug("Paper not exists in semantic scholar, arvixID : %s" % arvixID)
                continue
            authorIDList = [ int(author['authorId']) if author['authorId'] is not None else -1 for author in article_meta['authors']  ]
            authorNames = [article['main_author']]
            authorCount = len(article_meta['authors'])
            if authorCount > 1:
                other_author = [ name.strip() for name in article['authors'].split(',') if len(name) > 1 and name != article['main_author']]
                authorNames += other_author
            paper_category = [article['term']]
            if article['term'] != term:
                paper_category.append(term)
            try:
                paper = Paper.create(
                    indexID=idx+index,
                    arvixID = arvixID,
                    paperId = article_meta['paperId'],
                    doiID = str(article_meta['doi']),

                    title= article['title'],
                    summary=article['abstract'],
                    category=paper_category,
                    comments=article['comment'],
                    journal_ref=article['journal_ref'],

                    url=article['url'],
                    authorID=authorIDList,
                    authorName=authorNames,
                    authorCount=authorCount,
                    publishedDate=article['publish_date'],
                    citationVelocity=article_meta['citationVelocity'],
                    referencesCount=len(article_meta['references']),
                    topics=article_meta['topics'],
                    venue=str(article_meta['venue']),
                    year=article_meta['year'],
                    influentialCitationCount=article_meta['influentialCitationCount'],
                    citationCount=len(article_meta['citations']),
                    citations=article_meta['citations'],
                    )
                try:
                    for meta in ['page', 'figure', 'table']:
                        if meta in article['comment']:
                            comment = article['comment'].replace(';', ',')
                            for segment in comment.split(','):
                                if meta in segment:
                                    page_prefix = segment.split(meta)[0]
                                    if meta == 'page':
                                        paper.pages = int(page_prefix.strip())
                                    elif meta == 'figure':
                                        paper.figures = int(page_prefix.strip())
                                    elif meta == 'table':
                                        paper.table = int(page_prefix.strip())
                                    break
                except:
                    logging.debug("Error in parsing meta data")
                paper.save()
            except BaseException as e:
                logging.warning("Error in arvix id %s, error: %s" % (arvixID, str(e)))
            time.sleep(0.3)

def crawl_all():
    for term in tqdm(category_list):
        crawl_category(term=term)


def crawl_author():
    papers = Paper.select().execute()
    for paper in tqdm(papers):
        for author_id in paper.authorID:
            if author_id == -1:
                continue
            query = Author.select().where(Author.authorID == author_id)
            if query.exists():
                continue
            # try:
            success, author_profile = get_author_data(author_id)
            if success is False:
                logging.info("Author %d not exist!" % author_id)
                continue
            estCitation = author_profile['statistics']['estCitationAcceleration']['estimate']

            influencedIDList = author_profile['statistics']['influence']['influenced']
            influencedIDName = [ inf['author']['ids'][0] for inf in influencedIDList ]
            influenceCount = len(influencedIDName)

            influencedByIDList = author_profile['statistics']['influence']['influencedBy']
            influencedByIDName = [ inf['author']['ids'][0] for inf in influencedByIDList ]
            influenceByCount = len(influencedIDName)

            author = Author.create(
                authorID=author_id,
                name=author_profile['name'],
                semanticScholarUrl=author_profile['url'],

                hIndex=author_profile['statistics']['hIndex'],
                influentialCitationCount =author_profile['influentialCitationCount'],
                citationVelocity =author_profile['citationVelocity'],
                totalInfluentialCitationCount =author_profile['statistics']['totalInfluentialCitationCount'],
                maxEstCitationAcceleration=estCitation['max'],
                minEstCitationAcceleration=estCitation['min'],
                estCitationAcceleration=estCitation['value'],
                estCitationAccelerationConfidence=estCitation['confidence'],
                influencedIDList=influencedIDName,
                influencedPaper=influencedIDList,
                influenceCount=influenceCount,

                influencedByIDList=influencedByIDName,
                influencedByPaper=influencedByIDList,
                influenceByCount=influenceByCount,

                citationHistory=author_profile['statistics']['citedByYearHistogram'],
                totalPaper = len(author_profile['papers']),
            )
            author.save()
            time.sleep(0.3)
            # except BaseException as e:
            #     logging.warning("Error in author id %d: error %s" % (author_id, str(e)))


if __name__ == "__main__":
    update_paper()