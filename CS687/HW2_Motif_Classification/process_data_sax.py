import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import GetTime as gt
from ToSAX import ToSAX
import traceback
import os
from dask import delayed

def get_fns(full_rn=True):
    if full_rn:
        dataset1_test = 'hw1_datasets/dataset1/test_normalized.csv'
        dataset1_train = 'hw1_datasets/dataset1/train_normalized.csv'
        dataset2_test = 'hw1_datasets/dataset2/test_normalized.csv'
        dataset2_train = 'hw1_datasets/dataset2/train_normalized.csv'
        dataset3_test = 'hw1_datasets/dataset3/test_normalized.csv'
        dataset3_train = 'hw1_datasets/dataset3/train_normalized.csv'
        dataset4_test = 'hw1_datasets/dataset4/test_normalized.csv'
        dataset4_train = 'hw1_datasets/dataset4/train_normalized.csv'
        dataset5_test = 'hw1_datasets/dataset5/test_normalized.csv'
        dataset5_train = 'hw1_datasets/dataset5/train_normalized.csv'
        ds1_train_labels = 'hw1_datasets/dataset1/train_labels.csv'
        ds2_train_labels = 'hw1_datasets/dataset2/train_labels.csv'
        ds3_train_labels = 'hw1_datasets/dataset3/train_labels.csv'
        ds4_train_labels = 'hw1_datasets/dataset4/train_labels.csv'
        ds5_train_labels = 'hw1_datasets/dataset5/train_labels.csv'
        test_fns = [dataset1_test, dataset2_test, dataset3_test, dataset4_test, dataset5_test]
        train_fns = [dataset1_train, dataset2_train, dataset3_train, dataset4_train, dataset5_train]
        labels_fns = [ds1_train_labels, ds2_train_labels, ds3_train_labels, ds4_train_labels, ds5_train_labels]

    return test_fns, train_fns, labels_fns

def get_dataframes(verbose, fn_arr):
    if verbose:
        print('In get_dataframes: {}'.format(gt.time()))
    file_generator = (pd.read_csv(fn, index_col=0) for fn in fn_arr)
    return file_generator

def print_toSAX_csv(sax_df, i, window, nbins, letters, test_or_train, verbose=True):
    fn = 'hw1_datasets/dataset{}/{}_sax_w{}_nb{}_v{}.csv'.format(i, test_or_train, window, nbins, letters)
    print(fn)
    sax_df.to_csv(path_or_buf=fn)

def print_toDataset_csv(sax_df, i, window, nbins, letters, test_or_train, verbose=True):
    fn = 'hw1_datasets/dataset{}/{}_sax_w{}_nb{}_v{}.csv'.format(i, test_or_train, window, nbins, letters)
    sax_df.to_csv(path_or_buf=fn)

def run_sax_along_df(df, sax, as_string_vector=False, verbose=False):
    sax_strings = []
    for row in df.itertuples(name=None):  # (index, timeseries)
        ts = row[1:]
        if verbose and row[0]==0:
            print("number of obserbations is:{}\tthe window size is: {}\tExpected number of words is:{}"
                  .format(len(row[1:]),sax.window, len(row[1:])/sax.window))
        try:
            if as_string_vector:
                # symbolize_signal returns a list of words
                saxd_list = sax.symbolize_signal(ts)
                sax_string = " ".join(saxd_list) # convert list of sax words to a string of sax words, a sentence
                sax_strings.append(sax_string)
                # if verbose and row[0] % 10 == 0:
                #     print("time is:{};\tindex: {};\tnumber of records: {};\t{}% complete; ".
                #           format(gt.time(), row[0], len(df), round(row[0] / len(df), 2) * 100))

            else:
                sax_strings.append(sax.symbolize_signal(ts))
                # if verbose and row[0] % 10 == 0:
                #     print("time is:{};\tindex: {};\tnumber of records: {};\t{}% complete; ".
                #           format(gt.time(), row[0], len(df), round(row[0] / len(df), 2) * 100))
        except Exception:
            print('Exception:\n{}'.format(traceback.format_exc()))
            print("index:{};\tts:{}".format(row[0], row[1]))
    return pd.DataFrame(data=sax_strings) # either M x Number or words or M x 1 (each row is a string)

def grid_search_run(train_dfs, label_dfs, test_dfs, sax_words_as_svec, window_list, nbin_list, alphabet_list, verbose):
    dask_output = []
    i = 1
    for train_df, label_df, test_df in zip(train_dfs, label_dfs, test_dfs):
        for window in window_list:
            for nbins in nbin_list:
                for alphabet_size in alphabet_list:
                    s_time = gt.time()
                    if verbose:
                        print('dset: {}\tstime: {}\tw: {}\tnb: {}\ta: {}'.
                              format(i, gt.time(), window, nbins, alphabet_size))
                    alphabet = string.ascii_uppercase[:alphabet_size]
                    sax = ToSAX(window, nbins=nbins, alphabet=alphabet)
                    train_sax_df = run_sax_along_df(train_df, sax, sax_words_as_svec, verbose)
                    test_sax_df = run_sax_along_df(test_df, sax, sax_words_as_svec, verbose)
                    print_toSAX_csv(train_sax_df, i, window, nbins, alphabet_size, 'train')
                    print_toSAX_csv(test_sax_df, i, window, nbins, alphabet_size, 'test')
                    # train_sax_df = delayed(run_sax_along_df)(train_df, sax, sax_words_as_svec, verbose)
                    # test_sax_df = delayed(run_sax_along_df)(test_df, sax, sax_words_as_svec, verbose)
                    # dask_output.append(delayed(print_SAX_to_csv)(train_sax_df, window, nbins, alphabet_size, 'train'))
                    # dask_output.append(delayed(print_SAX_to_csv)(test_sax_df, window, nbins, alphabet_size, 'test'))

                    if verbose:
                        print('SAXed dataset: {}\tstart time: {}\tend time: {}'.format(i, s_time, gt.time()))
        i += 1

def regular_run(train_dfs, label_dfs, test_dfs, sax, sax_words_as_string_vec, window, nbins, alphabet_size, verbose):
    i=1
    for train_df, label_df, test_df in zip(train_dfs, label_dfs, test_dfs):
        s_time = gt.time()
        print(label_df.shape)
        # print(label_df)
        train_sax_df = run_sax_along_df(train_df, sax, sax_words_as_string_vec, verbose)
        test_sax_df = run_sax_along_df(test_df, sax, sax_words_as_string_vec, verbose)
        print_toSAX_csv(train_sax_df, i, window, nbins, alphabet_size, 'train')
        print_toSAX_csv(test_sax_df, i, window, nbins, alphabet_size, 'test')

        if verbose:
            print('SAXed dataset: {}\tstart time: {}\tend time: {}'.format(i, s_time, gt.time()))
            print('sax df shape: {}'.format(train_sax_df.shape))
        i += 1

def main():
    print(os.listdir())
    time = gt.GetTime()
    verbose = True
    full_run = True
    grid_search = True
    sax_words_as_string_vec = True  # scikit learn requires string vectors, does not accept 2d list
    #----------SAX--------------#
    window = 4
    stride = window
    nbins = 5
    alphabet_size = 5
    alphabet = string.ascii_uppercase[:alphabet_size]
    sax = ToSAX(window, stride, nbins, alphabet)
    #-----------GRID------------#
    window_list = range(5, 30, 5)
    nbin_list = range(5, 12, 2)
    alphabet_list = range(4, 9, 2)

    if verbose:
        print(time.initial_time)

    if full_run:
        test_fns, train_fns, labels_fns = get_fns(verbose)
        # dfs below are generators
        train_dfs = get_dataframes(verbose, train_fns)
        label_dfs = get_dataframes(verbose, labels_fns)
        test_dfs = get_dataframes(verbose, test_fns)

        if grid_search:
            if verbose: print('doing grid search')
            grid_search_run(train_dfs, label_dfs, test_dfs, sax_words_as_string_vec, window_list, nbin_list, alphabet_list, verbose)
        else:
            regular_run(train_dfs, label_dfs, test_dfs, sax, sax_words_as_string_vec, window, nbins, alphabet_size, verbose)
    else:
        if verbose: print('in else')
        i = 1
        dataset1_test = 'hw1_datasets/dataset4/test_normalized.csv'
        test1_labels = 'hw1_datasets/dataset4/test_labels.csv'
        dataset1_train = 'hw1_datasets/dataset4/train_normalized.csv'
        dataset1_train_labels = 'hw1_datasets/dataset4/train_labels.csv'
        train1 = pd.read_csv(dataset1_train, index_col=0)
        test1 = pd.read_csv(dataset1_test, index_col=0)
        train1_labels = pd.read_csv(dataset1_train_labels, index_col=0)

        # ts_sax_df = run_sax_along_df(train1.head(n=10), sax, sax_words_as_string_vec, verbose)
        train_sax_df = run_sax_along_df(train1, sax, sax_words_as_string_vec, verbose)
        test_sax_df = run_sax_along_df(test1, sax, sax_words_as_string_vec, verbose)
        if verbose:
            print('Results Found')
            print('sax df shape: {}'.format(train_sax_df.shape))
            # print(train_sax_df)
        print_toSAX_csv(train_sax_df, i, window, nbins, alphabet_size, 'train', True)
        print_toSAX_csv(test_sax_df, i, window, nbins, alphabet_size, 'test', True)

    if verbose:
        print('-------Completed!-------')
        print('Started at: {}\tFinished at: {}'.format(time.initial_time, gt.time()))

if __name__ == '__main__':
    main()