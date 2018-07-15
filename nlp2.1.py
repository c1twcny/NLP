#######################################################################
#
# NLP test code
#
# (1) Compute tf-idf for n documents
# (2) Compute cosine similarity matrix
#
# Ta-Wei Chen
#
# Date: July 1 2018
# Version 2.1
#
#######################################################################
import nltk
import string
import os
import collections
import sys
import logging
import math
import numpy as np
import scipy
import statsmodels
import pandas
import timeit

import matplotlib.pyplot as plt
import seaborn as sns

from os import listdir
from time import time
from optparse import OptionParser
from itertools import groupby
from string import punctuation
from collections import Counter
from collections import defaultdict
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from textblob import TextBlob

from nltk import RegexpTokenizer
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from scipy import spatial


# ----------------------------------------------------------------------
#
# functions
#
# ----------------------------------------------------------------------
def tokenize(input_file):
    my_tokenizer = RegexpTokenizer(r'\w+')
    newfile_no_punctuation = my_tokenizer.tokenize(input_file)
    newfile_no_punctuation = [word.strip() for word in
                              newfile_no_punctuation if word.lower()
                              not in my_stop_words]
    return newfile_no_punctuation


def stem_token(input_token, stemmer):
    stemmed_token = []

    for item in input_token:
        stemmed_token.append(stemmer.stem(item))
    return stemmed_token


def get_word_count(tokenized_file):
    count = Counter(tokenized_file)  # Counter is a subclass of Dict;
    # use myList.items() to access
    # {key:value} pair
    return count  # or count.most_common(10) to return the top 10


def normalized_term_frequency(local_word_counters_dict):
    total_occurrence = sum(local_word_counters_dict.values())
    for key, value in local_word_counters_dict.items():
        local_word_counters_dict[key] = float(value / total_occurrence)
    return local_word_counters_dict  # return dictionary with normalized
    #  frequency value for each key


def augmented_term_frequency(local_word_counters_dict):
    frequency_most_occurring_term = max(local_word_counters_dict.values())
    for key, value in local_word_counters_dict.items():
        local_word_counters_dict[key] = 0.5 + 0.5 * \
                                        float(value / frequency_most_occurring_term)
    return local_word_counters_dict  # return dictionary with
    # normalized frequency value for each key


def log_scalled_term_frequency(local_word_counters_dict):
    for key, value in local_word_counters_dict.items():
        local_word_counters_dict[key] = math.log10(float(1 + value))
    return local_word_counters_dict  # return dictionary with normalized
    #  frequency value for each key


def idf(num_of_docs, word_counters_dict):
    for key, value in word_counters_dict.items():
        word_counters_dict[key] = math.log10(num_of_docs / float(1 + value))
    return word_counters_dict


def convert_tokenfile_to_set(tokenized_file):
    return set(tokenized_file)  # eliminate duplicates but return random order list


def corpus_word_frequency(list_of_tokenized_file):
    temp_list = []
    temp_set = []
    for f in list_of_tokenized_file:
        temp_set.append(convert_tokenfile_to_set(f))
    for i in range(len(temp_set)):
        temp_list.extend(temp_set[i])
    return get_word_count(temp_list)


# ----------------------------------------------------------------------
#
# Parameter settings
#
# ----------------------------------------------------------------------
dir_path = '/home/u863713/PycharmProjects/NLP/nlp_data/'
# dir_path = '/home/u863713/Programming/python/nlp_data1/'
dir_stopword_path = '/home/u863713/PycharmProjects/NLP/mystopwords.txt'

with open(dir_stopword_path, 'r') as f:
    my_tmp = f.read().replace('\n', ' ')  # type: str
    my_tmp = re.sub(' +', ' ', my_tmp)
    additional_stop_words = my_tmp.split()

my_stop_words = stopwords.words('english') + list(punctuation) + additional_stop_words
stemmer = PorterStemmer()

file_name = []
results = []  # hold content of tokenized file
results_set = []  # hold non-duplicated token set from each tokenized file
tmp_dict = {}  # temporary dict for holding tf-idf output for each tokenized file
tf_list = []  # list holds tf output for all documents
tfidf_list = []  # list holds tf-idf output for all documents
tfidf_list_matrix = []
similarity_matrix = []
doc_similarity = []
tmp_file_list = []
word_count_list = []
pmi_list = []
pmi_matrix = []

np.set_printoptions(precision=4)  # set output precision of 4 digits

t0 = time()

# ----------------------------------------------------------------------
#
# (1) Read and clean files
#
# Output
#   file_name : type list
#               list of file names
#   tmp_file_list : type list
#               contents of each file stored in a list
#
# ----------------------------------------------------------------------
#
for my_file in listdir(dir_path):
    file_name.append(my_file)
    file_path = dir_path + my_file
    with open(file_path, 'r', encoding='utf8') as f:
        tmp_file = f.read().replace('\n', ' ')  # type: str
        tmp_file_list.append(tmp_file)

# ----------------------------------------------------------------------
#
# (1) Read and clean files
# (2) Tokenize each file
# (3) Store the content of tokenized files in a list
#
# Output
#   results : type list
#             total length of the list is the number of files
#             each element contains a list of tokenized strings
#
#             At this stage the 'results' list contains the raw data; rest
#             of the program performs various type of transformation &
#             processing on this raw data.
#
# ----------------------------------------------------------------------
#
for my_file in listdir(dir_path):
    file_path = dir_path + my_file
    with open(file_path, 'r', encoding='utf8') as f:
        file_content = f.read().replace('\n', ' ')  # remove a trailing newline
        # file_content = f.read().rstrip()
        file_content = file_content.strip().lower()  # remove leading & trailing white space; convert to lowercase
        #        file_content = ' '.join(file_content.split())   # remove extra white space between words
        file_content = re.sub(' +', ' ', file_content)
        tokenized_file = tokenize(file_content)  # tokenize strings
        tokenized_file = stem_token(tokenized_file, stemmer)
        results.append(tokenized_file)  # store content of tokenized file as an element in a list

# ----------------------------------------------------------------------
#
# (1) Create global corpus frequency dictionary
# (2) Calculate the idf value for each word across the global corpus
#
# Output
#   tmp_idf : type list
#             contain {word : idf} for all the corpus
#   ordered_idf_dict : type OrderedDict
#             output is an ordered {word : count} for all the corpus
#
#
# ----------------------------------------------------------------------
#
tmp_output = corpus_word_frequency(results)  # word frequency across all documents
tmp_idf = idf(len(results), tmp_output)  # idf of words across all documents
ordered_idf_dict = collections.OrderedDict(sorted(tmp_idf.items()))  # ordered dict based on key

# ----------------------------------------------------------------------
#
# (1) Create {word : count} dictionary for each word per document
# (2) Tally the total count for each word across ALL documents
# (3) Tally the total count for pairwise words (word-i, word-j) per document
# (4) Tally the results from (2) per pairwise words (word-i, word-j)
# (5) Place (4) in a square matrix
# (6) Divide square matrix from (5) by (2) for each word word-i to calculate PMI
# Pointwise Mutual Information for GLSA (General Latent Semantic Analysis) processing
#
#
# Output
#
#
#
# ----------------------------------------------------------------------
#
# Step 1
for f in results:
    tmp_wc_dict = collections.Counter(f)
    tmp_wc_dict_od = collections.OrderedDict(sorted(tmp_wc_dict.items(), key=lambda t: t[0]))
    word_count_list.append(tmp_wc_dict_od)
#    print(tmp_wc_dict_od)
print('\n')

# Step 2
# Based on https://stackoverflow.com/questions/19461747/sum-corresponding-elements-of-multiple-python-dictionaries
#
tmp_wc_corpus = collections.Counter()
for f in word_count_list:
    tmp_wc_dict.update(f)
tmp_wc_dict = collections.OrderedDict(sorted(tmp_wc_dict.items(), key=lambda t: t[0]))
new_tmp_wc_dict = collections.Counter()
new_tmp_wc_dict = {**new_tmp_wc_dict, **tmp_wc_dict}
new_tmp_wc_dict = dict.fromkeys(new_tmp_wc_dict, 0)
new_tmp_wc_dict = collections.OrderedDict(sorted(new_tmp_wc_dict.items(), key=lambda t: t[0]))
print('\n')
print(new_tmp_wc_dict)

# Step 3
# Step 3.1 Sum all the word occurrences for a document
for f in word_count_list:
    new_tmp_wc_dict = dict.fromkeys(new_tmp_wc_dict, 0)
    new_tmp_wc_dict = collections.OrderedDict(sorted(new_tmp_wc_dict.items(), key=lambda t: t[0]))
    #    print(new_tmp_wc_dict)
    wc_sum = sum(f.values())
    f.update((k, v / float(wc_sum)) for k, v in f.items())
    #    f = collections.OrderedDict(sorted(f.items(), key=lambda t: t[0]))
    f = {**new_tmp_wc_dict, **f}
    f = collections.OrderedDict(sorted(f.items(), key=lambda t: t[0]))

# Step 4

# Step 5

# Step 6


# ----------------------------------------------------------------------
#
# (1) Calculate the term-frequency (tf)
#
# Output
#   tf_list : type list
#             contain {word : tf} dictionary for each document
# ----------------------------------------------------------------------
#
for tokenized_file in results:
    file_word_count = get_word_count(tokenized_file)
    file_word_count = normalized_term_frequency(file_word_count)
    tf_list.append(file_word_count)

#    print(file_word_count)
#    for key, value in file_word_count.items():
#        print('{0} : {1:.6f}'.format(key, value))      # normalized term-frequency for each document
#    print('\n')

# ----------------------------------------------------------------------
#
# (1) Calculate the tf * idf values for words in each document
# (2) Store results in a list
#
# Output:
#   tfidf : type list
#           non-ordered {word : tf*idf} dictionary for each document
#           Length of dictionary varies
#   tfidf_list_matrix:
#           key-ordered {word : tf*idf} dictionary, with all the corpus,
#           for all documents.
#           Length of each dictionary is the same since all the corpus
#           are used
#  ----------------------------------------------------------------------
#
for f in tf_list:
    tmp_sharekeys = set()
    tmp_dict = {}
    tmp_sharekeys = set(f.keys()).intersection(tmp_idf.keys())
    for key in tmp_sharekeys:
        tmp_dict[key] = f[key] * tmp_idf[key]
    tfidf_list.append(tmp_dict)

tmp_output1 = ordered_idf_dict.copy()
init_full_corpus = dict.fromkeys(tmp_output1.keys(), 0.0)  # set default 0.0 for all the keys

for f in tfidf_list:
    merge_dict = {**init_full_corpus, **f}  # merge two dicts; 2nd dict values overwrite the 1st
    merge_dict_od = collections.OrderedDict(sorted(merge_dict.items()))
    tfidf_list_matrix.append(merge_dict_od)  # order the dict based on key

print('\n')

# ----------------------------------------------------------------------
#
# (1) Calculate the Cosine Similarity matrix between documents
#
# Output:
#   similarity_matrix : type list
#                       1 x n^2 vector of pair-wise similarity number
#                       for n documents
#   doc_similarity :
#                       n x n matrix of pair-wise similarity number
#                       for n documents
#
# ----------------------------------------------------------------------
#
for f in tfidf_list_matrix:
    od_val1 = list(f.values())
    for g in tfidf_list_matrix:
        od_val2 = list(g.values())
        similarity = 1.0 - spatial.distance.cosine(od_val1, od_val2)
        similarity_matrix.append(similarity)

# shape the similarity_matrix list into square matrix
s_row = len(tfidf_list_matrix)
s_col = int(len(similarity_matrix) / s_row)
doc_similarity = np.reshape(similarity_matrix, (s_row, s_col))
doc_sim = doc_similarity
s_file_name = [x.split('.')[0] for x in file_name]
doc_sim = pandas.DataFrame(doc_sim, columns=s_file_name)

# calculate the trimmed mean by excluding diagonal elements
similarity_matrix_subset = [x for x in similarity_matrix if x != 1.0]
trimmed_mean = sum(similarity_matrix_subset) / len(similarity_matrix_subset)
trimmed_max = max(similarity_matrix_subset)

# ----------------------------------------------------------------------
#
# (1) Calculate similarities using Latent Semantic Analysis
#     Note:
#
# Output:
#   xxxxx : type list
#                       1 x n^2 vector of pair-wise similarity number
#                       for n documents
#   xxxxx :
#                       n x n matrix of pair-wise similarity number
#                       for n documents
#
# ----------------------------------------------------------------------
#
n_row = len(tfidf_list_matrix)
n_col = len(tfidf_list_matrix[0])
test_matrix = []
for f in tfidf_list_matrix:
    od_val1 = list(f.values())
    test_matrix.append(od_val1)

test_matrix = np.asmatrix(test_matrix)
test_matrix_t = np.transpose(test_matrix)  # raw term-document matrix
# print(test_matrix_t.shape, np.linalg.matrix_rank(test_matrix_t))


# X = t x d matrix: t: number of terms, d: number of documents
# X = U * s * Vc
# Uc and Vc are the conjugate transpose matrix of U and V, respectively
# U is a t x t unitary matrix
# s is a diagonal t x d matrix; elements along the diagonal are singular values of X
# Vc is a d x d conjugate transpose matrix of V

U, s, Vc = np.linalg.svd(test_matrix_t, full_matrices=False)
Uc = np.matrix.getH(U)
V = np.matrix.getH(Vc)

print(U.shape, s.shape, Vc.shape)

sub_s = s[:len(s) - 0]
# s_square = np.matmul(np.transpose(np.asmatrix(s)), np.asmatrix(s))
# s_square = np.matmul(np.diag(s), np.diag(s))
s_square = np.dot(np.diag(sub_s), np.diag(sub_s))

s_size = sub_s.shape

sub_U = U[:, 0:s_size[0]]
sub_Vc = Vc[0:s_size[0], :]
sub_Uc = np.transpose(sub_U)
sub_V = np.transpose(sub_Vc)

X_matrix = np.dot(sub_U, np.dot(np.diag(sub_s), sub_Vc))
# Normalize column vector. In this case each column represents one document
# To normalize word occurrence across all documents, you need to set the axis paramenter to '1'
X_matrix_normalized = normalize(X_matrix, axis=0)  # normalize column vector - each column represents one document

# sub_Uc = np.matrix.getH(sub_U)
# sub_V  = np.matrix.getH(sub_Vc)

# reduced_s = np.asmatrix([i for i in s if i >= np.mean(s)])  # row matrix

# print(np.matmul(np.transpose(reduced_s), reduced_s)) # column matrix x row matrix

# Comparing two terms
# U * S^2 * Uc
# Using dot product between two row vectors of X
#
# word_vs_word = np.matmul(U*np.matmul(s, s), Uc) #original

# word_vs_word = np.matmul(sub_U, np.diag(s)) # webhome.cs.uvic.ca/~thomo/svd.pdf
# word_vs_word = np.matmul(np.matmul(sub_U, s_square), sub_Uc)
# word_vs_word = np.dot(sub_U, np.dot(np.diag(sub_s), np.transpose(sub_U)))
word_vs_word = np.dot(X_matrix_normalized, np.transpose(X_matrix_normalized))

# Comparing two documents
# V * S^2 * Vc
# Using dot product between two column vectors of X
#
# doc_vs_doc = np.matmul(V*np.matmul(s, s), Vc) #original

# doc_vs_doc = np.matmul(np.diag(s), sub_Vc) # webhome.cs.uvic.ca/~thomo/svd.pdf
# doc_vs_doc = np.matmul(sub_V, np.matmul(s_square, sub_Vc))
# doc_vs_doc = np.dot(np.transpose(sub_Vc), np.dot(np.diag(sub_s), sub_Vc))
doc_vs_doc = np.dot(np.transpose(X_matrix_normalized), X_matrix_normalized)
dd_max = np.matrix.max(np.asmatrix(doc_vs_doc))
dd_min = np.matrix.min(np.asmatrix(doc_vs_doc))
dd_median = np.median(np.asmatrix(doc_vs_doc).flatten(), axis=1)

# ----------------------------------------------------------------------
#
# Experiment: Using scikit-learn truncated SVD to perform LSA
#
# ----------------------------------------------------------------------
# svd = TruncatedSVD(n_components=s_size[0]-4)
# svd.fit_transform(test_matrix_t)
# print(svd.singular_values_)

tfidf_vector = TfidfVectorizer(tokenizer=tokenize, stop_words=my_stop_words, lowercase=True)
t_d_matrix = tfidf_vector.fit_transform(tmp_file_list)  # return tf-idf document-term sparse matrix
lsa_t_d_matrix = tfidf_vector.transform(tmp_file_list)  # return document-term sparse matrix [n_docs, n_terms]
svd = TruncatedSVD(n_components=s_size[0] - 4)  # perform truncated SVD;
svd.fit_transform(lsa_t_d_matrix)  # return matrix w/ reduced rank [n_docs, n_terms]
s_new = svd.singular_values_  # return singular values in 1-D array
sub_Vc_new = sub_Vc[0:s_new.shape[0], :]  # reduce row on Vc; note: X = U*s*Vc from previous SVD result
doc_vs_doc_new = np.matmul(np.diag(s_new), sub_Vc_new)  #

print('\n')
print('Processing time: %f s' % (time() - t0))
print('\n')
print(word_vs_word.shape, doc_vs_doc.shape)
# ----------------------------------------------------------------------
#
# Plot routines
#
# Python packages:
#   matplotlib
#   seaborn : statistical data visualization
#       dependency: pandas, numpy, scipy matplotlib, statsmodels
#
# Output:
#
# ----------------------------------------------------------------------
#


sns.set(style='white', font_scale=0.60)
# cmap = sns.palplot(sns.color_palette("Blues"))
# cmap = sns.palplot(sns.light_palette("green", n_color=8, reverse=True))
# color palette selections:
# YlGnBu, Blues, BuPu, Greens, coolwarm ...

# tf-idf/cosine-similarity
plt.figure()  # create new plot window
plt.title('Tf-idf & Cosine Similarity', fontsize=17)
sns.heatmap(doc_sim, cmap='YlGnBu', robust=True, linewidths=.1, vmin=0.0, vmax=trimmed_max, yticklabels=s_file_name)
# sns.heatmap(doc_sim, cmap='YlGnBu', robust=True, linewidths=.1, vmin=0.0, yticklabels=s_file_name)
# Latent Semantic analysis
# doc to doc comparison
plt.figure()  # create new plot window
plt.title('Latent Semantic Analysis', fontsize=17)
sns.heatmap(doc_vs_doc, cmap='YlGnBu', linewidths=.1, yticklabels=s_file_name, vmin=dd_min, vmax=0.25,
            xticklabels=s_file_name)

# Latent Semantic analysis
# word to word comparison
# plt.figure()
# sns.heatmap(word_vs_word, cmap='YlGnBu')


plt.figure()
sns.heatmap(doc_vs_doc_new, cmap='YlGnBu', linewidths=.1, vmin=0.0)

plt.show()  # only need to execute once

# od1 = collections.OrderedDict(sorted(tfidf_list_matrix[0].items()))
# od2 = collections.OrderedDict(sorted(tfidf_list_matrix[0].items()))
# ls_od1 = list(od1.values())
# ls_od2 = list(od2.values())
# similarity = 1 - spatial.distance.cosine(ls_od1, ls_od2)    # scipy spatial package
# print(similarity)


###
##tmp_output1 = ordered_idf_dict.copy()
# tmp_output1 = [{key : 0} for key in tmp_output1.keys()]
##tmp_output1 = dict.fromkeys(tmp_output1.keys(), 0.0)      # assign value 0 to every key
##z = {**tmp_output1, **tfidf_list[10]}                       # merge two dicts; 2nd dict's values overwrite the 1st ones
##ordered_z = collections.OrderedDict(sorted(z.items()))
##z_list = list(z.values())
##z_list.sort(reverse=True)
###
##print(z)
##print(ordered_z)
##print(type(tmp_output1))

# ----------------------------------------------------------------------
#
# Test code block
#
# ----------------------------------------------------------------------
#
tfidf_vector = TfidfVectorizer(tokenizer=tokenize, stop_words=my_stop_words, lowercase=True)
t_d_matrix = tfidf_vector.fit_transform(tmp_file_list)  # return tf-idf document-term sparse matrix
lsa_t_d_matrix = tfidf_vector.transform(tmp_file_list)  # return document-term sparse matrix [n_docs, n_terms]