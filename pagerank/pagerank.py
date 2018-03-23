import sys
import re
from operator import add
from pyspark import SparkContext

NUM_ITERATIONS = 20

def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split('\t', urls)
    return parts[0], parts[1]

if __name__ == "__main__":
#    if len(sys.argv) != 3:
#        print("Usage: pagerank <file> <iterations>", file=sys.stderr)
#        exit(-1)

    sc =SparkContext()
#    lines = sc.textFile(sys.argv[1],1)
    lines = sc.textFile('graph.txt', 2)

    # Loads all URLs from input file and initialize their neighbors.
    # the output file is of the format
    # (u'68', <pyspark.resultiterable.ResultIterable object at 0x7fcf0799b8d0>)
    # the second element is an interable object of all the links to that urls
    # e.g. for '68', >>> links.collect()[2][1].data outputs
    # [u'93', u'66', u'26', u'91', u'39', u'40', u'69', u'78', u'56', u'96', u'47']
    links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()

#===============================================================================
#                               Simple PageRank
#===============================================================================
    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # simple pagerank
    for iteration in range(NUM_ITERATIONS):
        # Calculates URL contributions to the rank of other URLs.
        # details: url_url_rank format is : an example below
        # (u'C', (<pyspark.resultiterable.ResultIterable object at 0x7fa017214b50>, 1.0))
        # thus in the lambda function, each url_urls_rank[1][0] is of the format :
        # <pyspark.resultiterable.ResultIterable object at 0x7fa017214b50>
        # url_urls_rank[1][1] = 1.0
        # computeContribs just divide 1 by the length of the iterable object
        contribs = links.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))
        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add)
    f = open('SimplePageRank.txt','w')
    for (link, rank) in ranks.collect():
        f.write(str(link) + '\t' + str(rank) + '\n')
    f.close()

#===============================================================================
#                           With spider traps
#===============================================================================

    # re-intialize ranks
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(NUM_ITERATIONS):
        # Calculates URL contributions to the rank of other URLs.
        # details: url_url_rank format is : an example below
        # (u'C', (<pyspark.resultiterable.ResultIterable object at 0x7fa017214b50>, 1.0))
        # thus in the lambda function, each url_urls_rank[1][0] is of the format :
        # <pyspark.resultiterable.ResultIterable object at 0x7fa017214b50>
        # url_urls_rank[1][1] = 1.0
        # computeContribs just divide 1 by the length of the iterable object
        contribs = links.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

        # Re-calculates URL ranks based on neighbor contributions.
        beta = 0.8
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * beta + (1-beta))

    # Collects all URL ranks and dump them to console.
    f = open('SpiderPageRank.txt','w')
    for (link, rank) in ranks.collect():
        f.write(str(link) + '\t' + str(rank) + '\n')
    f.close()
