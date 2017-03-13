import numpy as np
import pandas as pd
import string
from BagOfPatterns import BagOfPatterns
from ToSAX import ToSAX
import GetTime as gt
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, auc, f1_score

def get_fns(sax, full_rn=True):
    window, nbins, alphabet_size = sax.window, sax.nbins, sax.alphabet_cardinality
    test_fns = []
    train_fns = []
    train_label_fns = []
    test_label_fns = []

    for i in range(1, 6):
        data_test_fn = 'hw1_datasets/dataset{}/test_sax_w{}_nb{}_v{}.csv'.format(i, window, nbins, alphabet_size)
        data_train_fn = 'hw1_datasets/dataset{}/train_sax_w{}_nb{}_v{}.csv'.format(i, window, nbins, alphabet_size)
        train_labels = 'hw1_datasets/dataset{}/train_labels.csv'.format(i)
        test_labels = 'hw1_datasets/dataset{}/test_labels.csv'.format(i)
        train_fns.append(data_train_fn)
        test_fns.append(data_test_fn)
        train_label_fns.append(train_labels)
        test_label_fns.append(test_labels)

    return train_fns,test_fns, train_label_fns, test_label_fns

def get_dataframes(fn_arr, labels=False):
    if labels:
        print('in labels: {}'.format(fn_arr))
        file_generator = (pd.read_csv(fn, index_col=0).transpose().values.tolist()[0] for fn in fn_arr)
    else:
        print('in else: {}'.format(fn_arr))
        file_generator = (pd.read_csv(fn, index_col=0) for fn in fn_arr)
    return file_generator

def print_results_to_csv(predictions, dataset_num, sax, s_time, clf='kNN'):
    print('Printing Results')
    test_output_fn = 'test_output/results_dataset{}_{}_w{}_nb{}_a{}_s{}_e{}.csv'.format(
        dataset_num, clf, sax.window, sax.nbins, sax.alphabet_cardinality, s_time, gt.time())

    with open(test_output_fn, 'w') as results:
        for y in predictions:
            results.write('{0}\n'.format(y))


def main ():
    '''
    need to figure out what results I am going to test
    I could read in all predictions in folder
    should I only read a single dataset or all datasets?
    calculate metrics: f1, accuracy, number of classes, time, sax info
    move testoutput to different folder
    :return:
    '''

    ## first just grab one file and compare it to the labels
    ## will do dataset 1
    os.listdir(window=4,stride=4,nbins=5,alphabet_cardinality = 5)
    sax = ToSAX()




if __name__ == '__main__':
    main()
