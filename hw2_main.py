import numpy as np
import pandas as pd
import string
from BagOfPatterns import BagOfPatterns
from ToSAX import ToSAX
from sklearn.neighbors import KNeighborsClassifier
import GetTime as gt
from multiprocessing import cpu_count
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer


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

def run_classifiers(pipeline, params, gs, verbose=False):
    if dtw_run:
        dist_metric = 'dtw'

    else: dist_metric = 'euclidean'

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    if verbose: print('Fitting kNN')
    kNN.fit(train_df, train_labels, dist_metric, width, k)  #initializes kNN object

    return kNN.predict(test_df, parallel, nprocesses)

def cross_validation(clf, x, y, scoring_metric='f1', folds=5):
    cv_scores = cross_val_score(clf, x, y=y, groups=None, scoring=scoring_metric, cv=folds, n_jobs=-1, verbose=0,
                                fit_params=None,  pre_dispatch='2*n_jobs')
    # clf.score(trainx1,train_y1)

    return cv_scores

def grid_search_metrics(gs, hs=0):
    best_idx = np.argwhere(gs.cv_results_['rank_test_score'] == 1)
    best_test_score = float(gs_svc.cv_results_['mean_test_score'][int(best_idx)])
    if best_test_score > high_score - high_score*0.05:
        if best_test_score > high_score:
            best_flag = '$'
            high_score = best_test_score
            best_model = gs_svc.best_estimator_
            best_train_X, best_train_y = train_x, train_y
        else:
            best_flag = '*'

    best_test_std = float(gs.cv_results_['std_test_score'][int(best_idx)])
    best_params = gs.cv_results_['params'][int(best_idx)]

    k_search_result = (best_flag, train_x.shape, round(best_test_score, 6), round(best_test_std, 6), best_params)
    k_search_list.append(k_search_result)
    print(k_search_result)
    best_flag = ' '

def grid_search():
    # Grid search
    clf = RandomForestClassifier(n_estimators=1, n_jobs=-1)
    tuned_parameters = [{'n_estimators': [1,10,100,500],
                         'min_samples_split': [2,3,4],
                         'criterion': ["entropy"]},]

    gs = GridSearchCV(clf, tuned_parameters, cv=5, scoring='f1', n_jobs=-1)
    gs.fit(train_x,train_y)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

def main():

    #----------MAIN-------------#
    print(os.listdir())
    time = gt.GetTime()
    verbose = True
    full_run = True
    #----------SAX--------------#
    window = 4
    stride = window
    nbins = 5
    alphabet_size = 5
    alphabet = string.ascii_uppercase[:alphabet_size]
    sax = ToSAX(window, stride, nbins, alphabet)
    #-----------CLF------------#
    # grid_search = True
    k = 1
    njobs = cpu_count()-1

    if verbose:
        print(time.initial_time)

    if full_run:
        if verbose: print('in full run')
        train_fns, test_fns, train_label_fns, test_label_fns = get_fns(sax)

        # dfs below are generators
        train_dfs = get_dataframes(train_fns)
        train_label_dfs = get_dataframes(train_label_fns, True)
        test_label_dfs = get_dataframes(test_label_fns, True)
        test_dfs = get_dataframes(test_fns)

        results_array = []

        i = 1
        for train_sax, train_labels, test_sax, test_labels in zip(train_dfs, train_label_dfs, test_dfs, test_label_dfs):

            s_time = gt.time()
            print(test_labels)
            # create a Bag of patterns and transform sax representations to a tfidf matrix
            BoP = BagOfPatterns(train_sax, train_labels, test_sax, test_labels, sax)
            BoP.fit(sax_letter_cardinality=sax.alphabet_cardinality, do_tfidf=True, ngram_len=2, min_term_freq=1,
                    max_term_freq=1)
            train_tfidf, test_tfidf = BoP.transform()

            # classify tfidfs using kNN
            kNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='minkowski',
                                       p=2, n_jobs=njobs)
            kNN.fit(train_tfidf, train_labels)
            test_predicted = kNN.predict(test_tfidf)
            print_results_to_csv(test_predicted, i, sax, time.initial_time, 'kNN')

            if verbose: print('Results Found for dataset: {}\ttime: {}'.format(i, gt.time()))
            i += 1

    else:
        print('in else')
        i = 1
        data_test_fn = 'hw1_datasets/dataset{}/test_sax_w{}_nb{}_v{}.csv'.format(i, window, nbins, alphabet_size)
        data_train_fn = 'hw1_datasets/dataset{}/train_sax_w{}_nb{}_v{}.csv'.format(i, window, nbins, alphabet_size)
        test_labels = 'hw1_datasets/dataset{}/test_labels.csv'.format(i)
        dataset_train_labels = 'hw1_datasets/dataset{}/train_labels.csv'.format(i)
        dataset_test_labels = 'hw1_datasets/dataset{}/test_labels.csv'.format(i)
        x_sax = pd.read_csv(data_train_fn, index_col=0)
        test_sax = pd.read_csv(data_test_fn, index_col=0)

        # converting from pandas to list for scikit learn
        train_labels = pd.read_csv(dataset_train_labels, index_col=0).transpose().values.tolist()[0]
        test_labels = pd.read_csv(dataset_test_labels, index_col=0).transpose().values.tolist()[0]

        # create a Bag of patterns and transform sax representations to a tfidf matrix
        BoP = BagOfPatterns(x_sax, train_labels, test_sax, test_labels, sax)
        BoP.fit(sax_letter_cardinality=sax.alphabet_cardinality, do_tfidf=True, ngram_len=2, min_term_freq=1,
                max_term_freq=1)
        train_tfidf, test_tfidf = BoP.transform()

        # classify tfidfs using kNN
        kNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='minkowski', p=2, n_jobs=njobs)
        kNN.fit(train_tfidf, train_labels)
        test_predicted = kNN.predict(test_tfidf)

        print_results_to_csv(test_predicted, i, sax, time.initial_time, 'kNN')

        if verbose: print('Results Found')

    if verbose:
        print('-------Completed!{}-------')
        print('Started at: {}\tFinished at: {}'.format(time.initial_time, gt.time()))




        # knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='minkowski', n_jobs=-2).fit(r_df)
if __name__ == '__main__':
    main()

