#!usr/bin/python

import string
import csv
import re
import random
import binascii

# =============================================================================
#                Represent text as shingles, hash shingles integer
# =============================================================================

# Convert text to words, remove punctuations
def text_to_word(text):
	print '\nConvert sentences to words...'
	# input is a string
	text = text.translate(None, string.punctuation)
	text = text.split(' ')
	text = [x.replace('\r','') for x in text]
	text = [x.replace('\n','') for x in text]
	text = [x for x in text if x != '']
	text = [x.lower() for x in text]
	return text

# Convert words to ngrams
def word_to_ngrams(text):
	print '\nConvert words to ngrams...'
	ngram_list = []
	for i in range(1, len(text) -1):
		# create a string of shingle
		ngram_list.append(text[i-1]+ " " + text[i] + " " + text[i+1])
	return ngram_list

# Represent ngrams as integer, using crc32 as a hash function
def shingle_to_integer(ngram_list):
	print '\Hash ngrams to integers...'
	shinglesInDoc = set()
	for i in range(len(ngram_list)):
		shingle = ngram_list[i]
		crc = binascii.crc32(shingle) & 0xffffffff # convert the shigle to integer
		shinglesInDoc.add(crc)
	return shinglesInDoc  # return a set of integers representing that document

# =============================================================================
#                 		Generate MinHash functions
# =============================================================================

# Record the maximum shingle ID that we assigned.
maxShingleID = 2**32-1

# We need the next largest prime number above 'maxShingleID'.
# Looked this value up here:
# http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 4294967311

# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID.

# Generate a list of 'k' random coefficients for the random hash functions,
# while ensuring that the same value does not appear multiple times in the
# list.
def pick_random_coeffs(k):
	# Create a list of 'k' random values.
	randList = []
	while k > 0:
		# Get a random shingle ID
		randIndex = random.randint(0, maxShingleID)
		# Ensure that each random number is unique.
		while randIndex in randList:
			randIndex = random.randint(0, maxShingleID)
			# Add the random number to the list.
		randList.append(randIndex)
		k = k - 1
	return randList

# For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.

def generate_coeffs(numHashes):
	print '\nGenerating random MinHash functions...'
	coeffA = pick_random_coeffs(numHashes)
	coeffB = pick_random_coeffs(numHashes)
	return coeffA, coeffB


# =============================================================================
#                 	Generate MinHash signatures for all documents
# =============================================================================

# Rather than generating a random permutation of all possible shingles,
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index
# of the first shingle that you would have encountered in the random order.

# this function takes in a list of integers represented as shingles
# and list of coefficients for random permutation functions
# it outouts the signiture vector for a SINGLE Document
def min_hash_signature(shingleIDSet, coeffA, coeffB):
	print '\nGenerating MinHash signatures for all documents...'
	# The resulting minhash signature for this document.
	signature = []
	# For each of the random hash functions...
	for i in range(0, numHashes):
		minHashCode = nextPrime + 1
		# Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
		# the maximum possible value output by the hash.
		# For each shingle in the document...
		for shingleID in shingleIDSet:
			# Evaluate the hash function.
			hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime
			# Track the lowest hash code seen.
			if hashCode < minHashCode:
				minHashCode = hashCode
		# Add the smallest hash code value as component number 'i' of the signature.
		signature.append(minHashCode)
	signature = tuple(signature)
	return signature

# =============================================================================
#                 	Kmeans : Helper functions for math on list
# =============================================================================
import math # math needed for sqrt

# calculate Euclidean distances between two lists
def dist(p1, p2):
    elem_dist = [(p1[i] - p2[i]) **2 for i in range(len(p1))]
    distance = math.sqrt(sum(elem_dist))
    return distance

def sqrt_dist(p1, p2):
    sqrt_elem_dist = [(p1[i] - p2[i]) **2 for i in range(len(p1))]
    sqrt_distance = sum(sqrt_elem_dist)
    return sqrt_distance

# element-wise sum for tuple1 and tuple2
def tuple_element_sum(p1, p2):
    return tuple([p1[i] + p2[i] for i in range(len(p1))])

# element-wise division of tuple by integer v
def tuple_element_division(t, v):
    return tuple(t[i] / v for i in range(len(t)))


# =============================================================================
#        Label a signature by its cluster, returns the centroid
# =============================================================================
def label_by_cluster(signature, centroids):
	print '\nLabel data by cluster...'
	distances = []
	for i in range(len(centroids)):
		centroid = centroids[i]
		distance = dist(signature, centroid)
		distances.append(distance)
	index_min = min(xrange(len(distances)), key=distances.__getitem__)
	cluster = centroids[index_min]
	return cluster

def get_new_centroids(labeled_df):
	print '\nGet new centriods...'
	df_signature_cluster = labeled_df.map(lambda x: [x[0], x[1][1]]) # remove headings
	sum_dict = df_signature_cluster.reduceByKey(lambda x, y: tuple_element_sum(x, y)).collect()
	sum_dict = {cluster[0]: cluster[1] for cluster in sum_dict}
	count_dict = df_signature_cluster.countByKey()
	new_centroids = [tuple_element_division(sum_dict[k],float(count_dict[k])) for k in count_dict.keys()]
	return(new_centroids)


# =============================================================================
#       		    Data Preprocessing
# =============================================================================

print '\nRead data from csv...'
data = []
with open('articles.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		Article,Date,Heading,NewsType = row
		tup = (Heading, Article)
		data.append(tup)

print '\n Pass the data to Spark...'
data.pop(0)
df = sc.parallelize(data)

# Preprocessing
df = df.map(lambda x: (x[0], text_to_word(x[1])))
df = df.map(lambda x: (x[0], word_to_ngrams(x[1])))
df = df.map(lambda x: (x[0], shingle_to_integer(x[1])))

# MinHash documents
def minHash_data(df, numHashes):
	coeffA, coeffB = generate_coeffs(numHashes)
	df = df.map(lambda x: (x[0], min_hash_signature(x[1], coeffA, coeffB)))
	return df

# =============================================================================
#       		  				  K-means
# =============================================================================

# kmeans
def run_kMeans(df, k, MAX_Iterations):
	print '\nSample {} centroids from the RDD...'.format(k)
	sample_list = df.takeSample(False, k, 200)
	df.cache()
	centroids = [sample[1] for sample in sample_list]
	labeled_df = df.map(lambda x: [label_by_cluster(x[1], centroids), x])
	for i in range(1, MAX_Iterations):
		# get new centroids
		new_centroids = get_new_centroids(labeled_df)
		# update cluster
		labeled_df = df.map(lambda x: [label_by_cluster(x[1], new_centroids), x])
	return labeled_df

# SSE
def sse(labeled_df):
	print '\nCalculating total intra-cluster SSE...'
	df_signature_cluster = labeled_df.map(lambda x: [x[0], x[1][1]]) # remove headings
	df_element_variance = df_signature_cluster.map(lambda x: [1, sqrt_dist(x[0], x[1])]) # compute sqrt distance for point-cluster
	df_total_variance = df_element_variance.reduceByKey(lambda x, y: x+y)
	return df_total_variance.top(1)[0][1]

MAX_Iterations = 10
klist = range(1, 30)
# Test for optimal k and Iterations
numHashes = 100
df = minHash_data(df, numHashes)

# =============================================================================
#       		  				 SSE and plot
# =============================================================================

sse_list = []
for k in klist:
	labeled_df = run_kMeans(df, k, MAX_Iterations)
	sse_list.append(sse(labeled_df))

# import matplotlib.pyplot as plt
plt.plot(klist, sse_list, 'r--')
plt.xlabel('Number of clusters')
plt.ylabel('Intra-cluster variance')
plt.title('Length of signature = {}'.format(numHashes))
plt.show()

# =============================================================================
#       		  				 Check errors
# =============================================================================
# output the heading for all clusters
# signature = 100, k = 22.
df = sc.parallelize(data)
# Preprocessing
df = df.map(lambda x: (x[0], text_to_word(x[1])))
df = df.map(lambda x: (x[0], word_to_ngrams(x[1])))
df = df.map(lambda x: (x[0], shingle_to_integer(x[1])))

MAX_Iterations = 10
k = 22
numHashes = 100
df = minHash_data(df, numHashes)

# get labeled_df
labeled_df = run_kMeans(df, k, MAX_Iterations)
# get the list of shingles in df, then output the headings
labeled_df.top(1)[0][1][0]
labeled_df = labeled_df.map(lambda x: (x[0], x[1][0]))
all_labeled_headings = labeled_df.groupByKey().collect()

# Examine all headings
for i in range(k):
	print 'The headings for documents in cluster {} are: '.format(i)
	print [x[1].data for x in all_labeled_headings]

all_labeled_headings[5].data
