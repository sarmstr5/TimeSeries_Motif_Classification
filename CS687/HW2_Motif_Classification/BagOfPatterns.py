import pandas as pd
import numpy as np
import scipy.sparse as sp
import string
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer


class BagOfPatterns(object):
    '''
    Used to store SAX representations of time series and compare common motifs

    '''

    def __init__(self, train_df, train_labels, test_df, test_labels, sax):
        # assert isinstance(x_df, pd.DataFrame)
        # assert isinstance(labels, pd.DataFrame)

        # The sax words from the data forms the training corpus
        # Convert sax df to flattened list of strings for scikit learn
        print('this is train shape: {}'.format(train_df.shape))
        print('this is test shape: {}'.format(test_df.shape))
        self.train_sax = [" ".join(row) for row in train_df.values.tolist()]
        self.test_sax = [" ".join(row) for row in test_df.values.tolist()]
        #assert same lengths make sure length is right dim
        # assert len(self.train_sax) == len(self.train_labels)
        # assert len(self.test_sax) == len(self.test_labels)
        print(len(self.train_sax))
        print(len(self.test_sax))

        self.train_labels = train_labels
        self.test_labels = test_labels
        print(len(train_labels))
        print('Length of train labels: {}'.format(len(train_labels)))
        print('Length of test labels: {}'.format(len(test_labels)))

        self.classes = list(set(train_labels))

        self.sax = sax
        self.word_length = sax.nbins
        self.sax_letters = sax.alphabet

        self.vocab = None
        self.test_df = None
        self.verbose = None
        self.do_tfidf = None
        self.ngram_len = None
        self.min_term_freq=None
        self.max_term_freq=None
        self.train_transformed = None
        self.test_transformed = None

    def fit(self, sax_word_length=5, sax_letters=None, sax_letter_cardinality=5, do_tfidf=True, ngram_len=2,
            min_term_freq=1.0, max_term_freq=1.0, verbose=False):

        self.vocab = self.create_sax_vocab()
        self.verbose = verbose
        self.do_tfidf = do_tfidf
        self.ngram_len = ngram_len
        self.min_term_freq = min_term_freq
        self.max_term_freq = max_term_freq


    def create_sax_vocab(self):
        enumerated_vocab_list = []
        print(self.word_length)
        for enumd_letters in itertools.product(self.sax_letters, repeat=self.word_length):
            enumerated_vocab_list.append(''.join(enumd_letters))
        return enumerated_vocab_list

    def transform(self, do_tfidf=True):
        # sparse array with either tfidf or count occurances based on do_tfidf.
        # Decision affects possible classifiers
        if do_tfidf:
            print('performing tfidf')
            # Convert a collection of raw documents to a numpy CSR with TF-IDF features.
            vectorizer = TfidfVectorizer(strip_accents=None, ngram_range=self.ngram_len, lowercase=False,
                                         analyzer=str.split, vocabulary=self.vocab, min_df=self.min_term_freq,
                                         max_df=self.max_term_freq)
            self.train_transformed = vectorizer.fit_transform(self.train_sax, self.train_labels)   # tf-idf CSR np array
            self.test_transformed = vectorizer.fit_transform(self.test_sax, self.test_labels)   # tf-idf CSR np array

        else:
            print('creating token occurance counts')
            # Turns a collection of text documents into a scipy.sparse matrix holding token occurrence counts
            # (or binary occurrence information), possibly normalized as token frequencies if norm=’l1’
            # or projected on the euclidean unit sphere if norm=’l2’.
            # does not affect linearSVC(dual=True), Perceptron, SFDClassifier, PassiveAggressive)
            # affects linearSVC(dual=False), Lasso(), etc.
            hv = HashingVectorizer(strip_accents=False, stop_words=None, lowercase=False, analyzer=str.split)
            self.train_transformed = hv.fit_transform(self.train_sax, self.train_labels)
            self.test_transformed = hv.fit_transform(self.test_sax, self.test_labels)

        print('shape of TRAIN:{}\tshape of labels:{}\tshape of tfidf:{}'.
              format(len(self.train_sax), len(self.train_labels), self.train_transformed.shape))
        print('shape of TEST:{}\tshape of labels:{}\tshape of tfidf:{}'.
              format(len(self.test_sax), len(self.test_labels), self.test_transformed.shape))
        # self.x = transformed_x  #UPDATE TO FIX SELF.X how do i tell difference between test and train x? should I worry about it?

        return self.train_transformed, self.test_transformed

    def BoP_todisc(self, fn):
        np.savez(fn, data=self.x.data, indices=self.x.indices, indptr=self.x.indptr, shape=self.x.shape)


def load_BoP(fn):
    np_file = np.load(fn)
    return sp.csr_matrix(np_file['data'], np_file['indices'], np_file['indptr'], shape=np_file['shape'])
