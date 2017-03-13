import numpy as np
import pandas as pd
from fractions import Fraction
from string import ascii_uppercase
import dask.dataframe as dd
from dask import visualize, delayed
# from distributed import Client, progress
# import joblib


class ToSAX(object):
    '''
    Symbolic Aggregate Approximation
    main usage for time series data:
        indexing and query
        calculating distance between time-sereis and thus perform clustering/classification
        symbolic representation for time series - inspiring textmining related tasks such as association mining
        vector representation of time-series

    algorithm steps:
        1) Segment time-series data into gapless pieces (e.g., gap introduced by missing values or change of sampling frequences)
        2) Each piece will be SAXed into a sequence of "words" (e.g., "abcdd" "aabcd", ...).
        3) This is done by rolling a sliding window of length 'window' with a stride of length 'stride'.
            If stridestride < window, there will be overlapping of different windows.
        4) Each window will be converted into one word

        for each sliding window:
        1 normalize across the window #not needed the data being read in is already normalized
        2 Convert to discrete values on time axis (index) by grouping points into equal-sized bins
            (bin sizes can be fractional, weighs each bin proportionally) - controlled by nbins. For each bin,
            use the mean of bin as local approximation.
        3 discretize on value axis by dividing values into nlevels quantiles  (equiprobability assuming standard normal data),
            for each level, calculate the "letter" by cutpoint table
        4 at the end, each bin in a sliding window will be mapped to a letter,
            each window in the piece of time-series will be mapped to a word,
            and the whole piece of series will be a sentence
        5 calcualte the distance between two symoblic representations by their
            corresponding levels
        6 if a vector representation is necessary, each letter can be mapped to a scalar value, such as the mean of the corresponding level.
    '''
    def __init__(self, window=None, stride=None, nbins=None, alphabet="ABCDE", alphabet_cardinality = None, verbose=False):
        '''
        Algorithim based on the paper "Experiencing SAX: a Novel Symbolic Representation of Time Series" by Lin, Keogh, Wei, and Lonardi
        Code based from implementations of SAX, pysax.py, SAX_2006, and saxpy.py
        Assume a gapless (fixed freq. no missing value) time series  #### update this
        window: sliding window length to define the number of words
        stride: stride of sliding, if stride < window, there is overlapping in windows
        nbins: number of bins in each sliding window, defining the length of word
        alphabet: alphabet for symbolization, also determines number of value levels
        cutputs: partition points for alphabet, equal to cardinality of alphabet
        # Not all parameters are used if only partial functions of the class is needed
        '''

        self.window = window
        self.stride = (stride or window)
        self.nbins = nbins
        self.alphabet = list(alphabet)
        self.alphabet_cardinality = alphabet_cardinality or len(alphabet)
        if (len(alphabet) != alphabet_cardinality) and (alphabet_cardinality is not None):
            print('alphabet and alphabet cardinality do not match, updating based on cardinality')
            self.alphabet = ascii_uppercase[:alphabet_cardinality]

        self.nlevels = self.alphabet_cardinality
        self.verbose = verbose
        # self.sax_length = sax_length
        self.cut_points_dict = {3 : [-np.inf, -0.43, 0.43, np.inf],
                                4 : [-np.inf, -0.67, 0, 0.67, np.inf],
                                5 : [-np.inf, -0.84, -0.25, 0.25, 0.84, np.inf],
                                6 : [-np.inf, -0.97, -0.43, 0, 0.43, 0.97, np.inf],
                                7 : [-np.inf, -1.07, -0.57, -0.18, 0.18, 0.57, 1.07, np.inf],
                                8 : [-np.inf, -1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, np.inf],
                                9 : [-np.inf, -1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, np.inf],
                                10: [-np.inf, -1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28, np.inf],
                                11: [-np.inf, -1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34, np.inf],
                                12: [-np.inf, -1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38, np.inf],
                                13: [-np.inf, -1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43, np.inf],
                                14: [-np.inf, -1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47, np.inf],
                                15: [-np.inf, -1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5, np.inf],
                                16: [-np.inf, -1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53, np.inf],
                                17: [-np.inf, -1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56, np.inf],
                                18: [-np.inf, -1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59, np.inf],
                                19: [-np.inf, -1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62, np.inf],
                                20: [-np.inf, -1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, np.inf]
                                }
        # different cut points derived from the alphabet length
        self.cut_points = self.cut_points_dict[len(self.alphabet)]
        if verbose:
            print("initializing")
            print("The window is: {}".format(self.window))
            print("The stride is: {}".format(self.stride))
            print("The number of bins is: {}".format(self.nbins))
            print("The alphabel is: {}".format(self.alphabet))
            print("The number of levels is: {}".format(self.nlevels))
            print("The cut points are: {}".format(self.cut_points))
            print("\n-------------------------\n")
        # grabbing the median because the mean can be unreliable
        vecs = [(a+b)/2 for (a, b) in zip(self.cut_points, self.cut_points[1:])]
        # replacing first and last values
        vecs[0] = self.cut_points[1]
        vecs[-1] = self.cut_points[-2]
        self.sym2vec = dict(zip(self.alphabet, vecs))

    def symbolize_window(self, window_data):
        """
        Step 0.0 - Convert time series to sympolic representation
        Symbolize one sliding window signal to a word
        """
        if self.verbose:
            print("In 'symbolize_window'")
        paa = self.paa_window(window_data)
        return "".join(self.symbolize(paa)) #concatenates string

    def symbolize(self, xs):
        """
        Symbolize as a PPA
        """
        if self.verbose:
            print("In 'symbolize'")
        alphabet_sz = len(self.alphabet)
        cutpoints = self.cut_points_dict[alphabet_sz]
        if self.verbose:
            print('the cutpoints are: {}'.format(cutpoints))
        return pd.cut(xs, bins = cutpoints, labels = self.alphabet)

    def paa_window(self, window_data):
        """
        1.0 - Convert time series into PAA
        piecewise aggregate approximation: one sliding window signal to a word
        abstracts time series data points to medians
        """
        if self.verbose:
            print("In 'paa_window'")
        window_data = self.normalize(window_data) #data already normalized
        binsize = Fraction(len(window_data), self.nbins)  # stores numerator/denominator object
        xs = map(lambda ss: np.sum(ss) / float(binsize), self.binpack(window_data))
        return [mapping for mapping in xs]

    def normalize(self, window_signal):
        """
        1.1 - Normalize for PAA step
        Normalize the data, if it hasnt been done
        it should be local to a sliding window
        """
        if self.verbose:
            print("In 'normalization'")

        s = np.asarray(window_signal)
        mu, sd = np.mean(s), np.std(s)
        return (s - mu) / (sd + 1e-10)

    def binpack(self, xs):
        """
        1.2 - discretize on time axis (index) by grouping points into equal-sized bins
        (bin sizes could be fractional) - controlled by nbins.
        For each bin, use the mean of bin as local approximation.
        for a singal of length 5, nbins = 3,
        it generates (p1, 2*p2/3), (p2/3, p3, p4/3), (2*p4/3, p5)
        """
        if self.verbose:
            print("In 'binpack'")

        # normalized time series data
        xs = np.asarray(xs)
        #converts to fraction object, length of
        binsize = Fraction(len(xs), self.nbins)
        # weighs each bin uniformly based on number of bins and (length of input/number of bins)
        wts = [1 for i in range(int(binsize))] + [binsize-int(binsize)]
        pos = 0
        while pos < len(xs):
            n = len(wts) - 1 if wts[-1] == 0 else len(wts)
            yield xs[pos:(pos+n)] * wts[:n]
            pos += len(wts) - 1
            rest_wts = binsize-(1-wts[-1])
            wts = [1-wts[-1]] + [1 for i in range(int(rest_wts))] + [rest_wts-int(rest_wts)]

    def sliding_window_index(self, signal_length):
        """
        1.3
        Takes length of signal and returns generator of indices
        each indice defines a sliding window
        """
        if self.verbose:
            print("In 'sliding_window_index'")
            print('window size: {}'.format(self.window))
            print('signal size: {}'.format(signal_length))
        start = 0
        while (start + self.window) <= signal_length:
            yield slice(start, start+self.window)
            start += self.stride

    def symbolize_signal(self, signal, parallel=None, n_jobs = -1):
        """
        Symbolize whole time-series signal to a sentence (vector of words),
        parallel can be {None, "joblib"}
        returns list of words
        """
        if self.verbose:
            print("In 'symbolize_signal'")
            print('signal is:\n{}'.format(signal))

            # generator of n windows as defined during instantiation
        window_index = self.sliding_window_index(len(signal))

        # Nonparallel solution
        # Calls symbolize window, converts each window to a word
        # returns a list of words
        if parallel == None:
            if self.verbose:
                print("Not parallel")
            return list(map(lambda wi: self.symbolize_window(signal[wi]), window_index))

    def symbol_to_vector(self, words):
        if self.verbose:
            print("In 'symbol_to_vector'")
        return np.array([np.asarray([self.sym2vec[w] for w in word]) for word in words])


    def min_dist(self, word1, word2):
        if self.verbose:
            print("In 'symbol_distance'")
        cutpoints = self.cut_points
        inverted_alphabet = dict([(w,i) for (i,w) in enumerate(self.alphabet, 1)])
        diff = np.asarray([0 if abs(iw1-iw2) <= 1 else cutpoints[max(iw1,iw2)-1] - cutpoints[min(iw1, iw2)]
                           for (iw1, iw2) in zip(map(inverted_alphabet.get, word1), map(inverted_alphabet.get, word2))])
        return np.sqrt(np.sum(diff**2))

    def convert_index(self, word_indices = None, ts_indices = None):
        '''
        if word_index is not None, convert word (sliding window) index to time series index
        otherwise convert ts_index to word_index
        '''
        if self.verbose:
            print("In 'convert_index'")

        if word_indices is not None:
            return [wi * self.stride for wi in word_indices]
        elif ts_indices is not None:
            return [int(ti / self.stride) for ti in ts_indices]
        else:
            raise ValueError("either word_index or ts_index needs to be specified")

def main():
    x = np.random.randint(0,50,50)
    print(x)
    print(len(x))
    sax = ToSAX(5,5,3,'ABCD',True)
    x_sax = sax.symbolize_signal(x)
    print(x_sax)

if __name__ == '__main__':
    main()

  # def run_SAX_along_channels(self, full_df, flatten=False):
    #     '''
    #     helper function to run SAX along the 16 channels in the dataframe
    #     removes observation if all channels have NaN at a given instance
    #     '''
    #     df = self.remove_NaNs(full_df)
    #     if df.empty:
    #         return pd.DataFrame([np.NaN])
    #     if flatten:
    #         flat_x = full_df.values.flatten().tolist()  #flattened dataframe
    #         # standardizing SAX string represenation size in preparation of proximity comparisons
    #         # necessary because of removed columns and rows
    #         #         window_size = int(ceil((len(flat_x)*1.0)/sax_symbol_len)) # each time series SAX string is the same length
    #         window_size = int(ceil((len(flat_x)/self.sax_length))) # each time series SAX string is the same length'
    #         self.window = window_size
    #         self.stride = window_size
    #         # find SAX representation for each column
    #         saxd_list = self.symbolize_signal(flat_x)
    #         #         if self.verbose:
    #         print('window_size: {}'.format(window_size))
    #         print('length of x is {}'.format(len(flat_x)))
    #         print('len of sax list is {}'.format(len(saxd_list)))
    #         return "".join(saxd_list) # convert list of strings to a string
    #
    #     else: # use for tf-idf
    #         #             print('in else')
    #         col_list = []
    #         for col in df:
    #             print('iterating on channels in channel: {}'.format(col))
    #             channel = df[col].tolist()
    #             saxd_channel = self.symbolize_signal(channel)
    #             #                 print(len(channel))
    #             col_list.append(saxd_channel)
    #             # convert list of lists to dictionary and create row ids to create pandas df
    #         #             channels_dict = {'SAXch{:0>2}'.format(i): col_list[i] for i in range(0, len(col_list))}
    #         #             row_idxs = ['word{:0>2}'.format(i) for i in range(0, len(col_list))]
    #         #             return pd.DataFrame(channels_dict)
    #         sax_string_list = [sax_string for col in col_list for sax_string in col]
    #
    #         return pd.DataFrame(data=sax_string_list).T
    #
    # def remove_NaNs(self, df):
    #     #remove NaN rows
    #     mask = df.notnull().all(axis=1)
    #     trimmed_rows_df = df[mask]
    #
    #     #remove NaN cols
    #     mask = trimmed_rows_df.notnull().all(axis=0)
    #     trimmed_cols_df = trimmed_rows_df.loc[:,mask]
    #
    #     # fill in left over NaN elements with previous elements
    #     cleaned_df = trimmed_cols_df.fillna(method='ffill')
    #     return cleaned_df
