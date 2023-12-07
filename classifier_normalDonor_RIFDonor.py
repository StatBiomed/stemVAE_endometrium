# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：classifier_normalDonor_RIFDonor.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/21 10:26

geneset: 2000-hvg
cellset: 02-major

disease classification
building classifer to discriminate RIF and normal and to identify the most important genes contributing to RIF

the 10 RIF samples were firstly classified as two categories: one is that demonstrated predicted time shift;
the remaining are that with no significant shift

normal: six LH7 donors; RIF class 1: RIF_4, RIF_8, RIF_10; RIF class 2: the remaining 7 RIF samples.
"""
import numpy as np
import time
import scanpy as sc
import anndata

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
    average_precision_score,
    matthews_corrcoef,
)
import pandas as pd
import scanpy as sc

from utils.logging_system import  LogHelper
import logging
from collections import Counter
import random

global_random_seed = 0
random.seed(global_random_seed)
print("set global random seed: {}".format(global_random_seed))


def main(min_cell_num=50, min_gene_num=100, RIF_donor_list=["RIF_4", "RIF_8", "RIF_10"], normal_donor_list=None,
         special_path_str=""):
    save_path = "results/231207_calssifier_normalDonor_RIFDonor_tree100_07fibro/{}/".format(special_path_str)
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger_file = 'logs/classifier_normalDonor_RIFDonor.log'

    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    # -----------preprocess data---------
    adata = anndata.read_csv("data/preprocess_07_fibro_Anno0717_Gene0720/data_count_hvg.csv",delimiter='\t')
    adata = adata.T  # 基因和cell转置矩阵
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    cell_time = pd.read_csv("data/preprocess_02_major_Anno0717_Gene0720/cell_with_time.csv", sep="\t",index_col=0)
    donor_list = np.unique(cell_time["donor"])
    if normal_donor_list is None:
        normal_donor_list = [_d for _d in donor_list if _d[:3] == "LH7"]
    # RIF_donor_list = [_d for _d in donor_list if _d[:3] == "RIF"]
    donor_list = normal_donor_list + RIF_donor_list

    save_cell_list = list(set(cell_time.loc[cell_time["donor"].isin(donor_list)].index) & set(adata.obs.index))
    adata = adata[save_cell_list].copy()
    cell_time = cell_time.loc[save_cell_list]
    _logger.info("Donor list: {}\nDetail: {}".format(donor_list, Counter(cell_time["donor"])))
    _logger.info("adata shape: {}".format(adata.shape))
    # 数据数目统计
    _shape = adata.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata.shape
    _logger.info("After drop gene threshold: {}, cell threshold: {}, remain adata shape: {}".format(min_cell_num,
                                                                                                    min_gene_num,
                                                                                                    adata.shape))
    cell_time = cell_time.loc[adata.obs.index]
    _logger.info("Donor list: {}\nDetail: {}".format(donor_list, Counter(cell_time["donor"])))

    from utils.utils_plot import  plot_boxPlot_nonExpGene_percentage_whilePreprocess
    plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, "donor", "", "",
                                                       special_file_str="", save_images=False)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    _logger.info("Finish normalize per cell, so that every cell has the same total count after normalization.")

    sc_expression_df = pd.DataFrame(data=adata.X, columns=adata.var.index, index=adata.obs.index)

    denseM = sc_expression_df.values
    # denseM = KNN_smoothing_start(denseM, type=KNN_smooth_type)
    # _logger.info("Finish smooth by {} method.".format(KNN_smooth_type))
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    _logger.info("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=sc_expression_df.columns, index=sc_expression_df.index)

    # make label, 0 for LH7, 1 for RIF
    normal_donor_cell_list = list(set(cell_time.loc[cell_time["donor"].isin(normal_donor_list)].index) & set(adata.obs.index))
    RIF_donor_cell_list = list(set(cell_time.loc[cell_time["donor"].isin(RIF_donor_list)].index) & set(adata.obs.index))
    _logger.info("normal donor number: {}, RIF donor number: {}".format(len(normal_donor_cell_list), len(RIF_donor_cell_list)))
    normal_donor_cell_label_df = pd.DataFrame(data=np.zeros(len(normal_donor_cell_list)).astype(int),
                                              index=normal_donor_cell_list, columns=["label"])
    RIF_donor_cell_label_df = pd.DataFrame(data=1 + np.zeros(len(RIF_donor_cell_list)).astype(int), index=RIF_donor_cell_list,
                                           columns=["label"])
    sc_label_df = pd.concat([normal_donor_cell_label_df, RIF_donor_cell_label_df])
    sc_label_df = sc_label_df.reindex(sc_expression_df.index)

    # feature selection to select important genes
    # from pyHSICLasso import HSICLasso
    # hsic_lasso = HSICLasso()
    # hsic_lasso.input(np.array(sc_expression_df.values),
    #                  np.array(sc_label_df["label"].values),
    #                  featname=sc_expression_df.columns)
    # hsic_lasso.classification(num_feat=100, n_jobs=-1, B=75)
    #
    # hsic_lasso.dump()
    # top100_gene_list=hsic_lasso.get_features()

    # split sc data into train set and test set
    train_x_df, train_y_df, test_x_df, test_y_df, randomSeed = read_data(sc_expression_df, sc_label_df, train_ratio=0.8)
    setting = {"classifiers": ["RF_sig","RF"]}
    clf_result_dic = classifiers([i for i in range(len(sc_expression_df.columns))], trainX_df=train_x_df, trainY_df=train_y_df,
                                 testX_df=test_x_df,
                                 testY_df=test_y_df, setting=setting, save_path=save_path)
    # clf_result_dic["top100_gene_list"]=top100_gene_list
    clf_result_dic["gene_symbol"] = sc_expression_df.columns
    _logger.info("for RIF list {}, \n classification results includes: {}".format(RIF_donor_list, clf_result_dic.keys()))

    save_file_name = "{}/clf_results.json".format(save_path)
    import json
    with open(save_file_name, 'w') as f:
        json.dump(clf_result_dic, f, default=str)  # 2023-07-03 22:31:50
    _logger.info("Finish save clf result at: {}".format(save_file_name))
    return


# split data into train and test data by ratio. ratio=dom(train data)/dom(data)
def read_data(sc_df, label_df, randomseed=global_random_seed, train_ratio=0.8):
    print("split the data as {} as train data and {} as test data".format(train_ratio, 1 - train_ratio))

    df_with_labels = pd.concat([sc_df, label_df], axis=1)
    df_shuffled = df_with_labels.sample(frac=1, random_state=randomseed)  # 使用随机种子保证结果可重复

    train_size = int(train_ratio * len(df_shuffled))
    train_df = df_shuffled[:train_size]
    test_df = df_shuffled[train_size:]

    while len(np.unique(train_df["label"])) != len(np.unique(test_df["label"])):
        print("random_seed needs to be changed as ", randomseed)
        return read_data(sc_df, label_df, randomseed=randomseed + 10, train_ratio=train_ratio)
    column_gene_list = [col for col in train_df.columns if col != "label"]
    return train_df[column_gene_list], train_df["label"], test_df[column_gene_list], test_df["label"], randomseed


def acc_score(label, y_pred):
    return accuracy_score(label, y_pred)


def auc_roc(label, y_pred):
    return roc_auc_score(label, y_pred)


def auc_prc(label, y_pred):
    '''Compute AUCPRC score.'''
    return average_precision_score(label, y_pred)


def f1_optim(label, y_pred):
    '''Compute optimal F1 score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    f1s = 2 * (prec * reca) / (prec + reca)
    return max(f1s)


def gm_optim(label, y_pred):
    '''Compute optimal G-mean score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    gms = np.power((prec * reca), 0.5)
    return max(gms)


def mcc_optim(label, y_pred):
    '''Compute MCC score.'''
    lable_num = len(np.unique(label))
    if len(np.unique(y_pred)) != lable_num:
        return 0
    mcc = matthews_corrcoef(label, y_pred)
    return mcc


def plot_hist_prob(predict_probs, testY, classifier, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    class_0_0 = predict_probs[testY == 0][:, 0]  # 类别0的概率
    class_1_1 = predict_probs[testY == 1][:, 1]  # 类别1的概率
    plt.hist([class_0_0, class_1_1], bins=20, color=['blue', 'orange'], alpha=0.7,
             label=['Class 0 to 0', 'Class 1 to 1'])
    class_0_1 = predict_probs[testY == 0][:, 1]  # 类别0的概率
    class_1_0 = predict_probs[testY == 1][:, 0]  # 类别1的概率
    plt.hist([class_0_1, class_1_0], bins=20, color=['green', 'red'], alpha=0.7,
             label=['Class 0 to 1', 'Class 1 to 0'])
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('True Label Probability Distribution')
    plt.legend()
    plt.savefig('{}/{}predict_onTest.png'.format(save_path, classifier))
    plt.show()


def classifiers(features, trainX_df, trainY_df, testX_df, testY_df, setting, labelNum=2, save_path=""):
    # classify
    trainX = trainX_df.values
    testX = testX_df.values
    trainY = trainY_df.values
    testY = testY_df.values
    train_x = trainX[:, features]
    test_x = testX[:, features]
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'RF_sig': random_forest_classifier_sig,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier,
                   'GBDT-feature-selection': GBDT_feature_selection_classifier
                   }

    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

    result = dict()
    for classifier in setting["classifiers"]:
        print('******************* %s ********************' % classifier)

        print(train_x.shape)
        start_time = time.time()
        model = classifiers[classifier](train_x, trainY)
        print(classifier)
        print('training took %.2fs!' % (time.time() - start_time))
        # if classifier=="RF_sig":

        predict = model.predict(test_x)
        predict_probs = model.predict_proba(test_x)
        plot_hist_prob(predict_probs, testY, classifier, save_path)
        # 绘制柱状图表示每个样本在真实标签上的概率分布
        try:
            feature_importance=model.feature_importances_
        except:
            try:
                feature_importance=model.estimator.feature_importances_
            except:
                feature_importance=None
        if (labelNum == 2):
            ##mark need to change
            aucroc = auc_roc(testY, predict)
            acc = acc_score(testY, predict)
            aucprc = auc_prc(testY, predict)
            f1 = f1_optim(testY, predict)
            gm = gm_optim(testY, predict)
            mcc = mcc_optim(testY, predict)
            from sklearn.metrics import log_loss
            score = log_loss(testY, predict_probs)

            result[classifier] = {"acc": acc, "aucroc": aucroc, "aucprc": aucprc,
                                  "f1": f1, "gm": gm, "mcc": mcc,
                                  "score": score,
                                  "cell_id": list(testY_df.index), "cell_pred": list(predict),
                                  "cell_pred_probs":predict_probs,
                                  "cell_real": list(testY_df.values)}
            if feature_importance is not None:
                result[classifier]["feature_importance(moreHigerMoreImportant)"]= feature_importance
            print("mark 2023-08-23 15:21:04", classifier, acc, aucroc)
        # for muli class
        else:
            print("Label should be binary")
    return result


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=global_random_seed)
    model.fit(train_x, train_y)
    return model


def random_forest_classifier_sig(train_x, train_y):
    # train_x_df, train_y_df, test_x_df, test_y_df, randomSeed = read_data(sc_expression_df, sc_label_df,train_ratio=0.8)
    from sklearn.model_selection import train_test_split
    train_x, valid_x, train_y, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=global_random_seed)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=global_random_seed)
    model.fit(train_x, train_y)
    sig_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    sig_model.fit(valid_x, y_valid)
    # sig_clf_probs = sig_model.predict_proba(X_test)
    # sig_score = log_loss(y_test, sig_clf_probs)
    return sig_model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier,but classifer raw data without any
# feature selection method, and feature selection by GBDT feature_importances.
def GBDT_feature_selection_classifier(train_x, train_y, raw_feature, selection_feature_number):
    from sklearn.ensemble import GradientBoostingClassifier
    import heapq

    model = GradientBoostingClassifier(n_estimators=200, verbose=1)
    model.fit(train_x, train_y)
    # calculate feature importance
    a = model.feature_importances_
    w = heapq.nlargest(selection_feature_number, range(len(a)), a.take)
    pram = [[raw_feature[i], a[i]] for i in w]

    np.savetxt("feature_importances_by_GBDT_rawdata.csv", pram, delimiter=',', fmt="%s")

    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    RIF_donor_list1 = ["RIF_4", "RIF_8", "RIF_10"]
    RIF_donor_list2 = ['RIF_1', 'RIF_2', 'RIF_3', 'RIF_5', 'RIF_6', 'RIF_7', 'RIF_9']

    main(RIF_donor_list=RIF_donor_list1, special_path_str="normalDonor_VS_RIFClass1")
    main(RIF_donor_list=RIF_donor_list2, special_path_str="normalDonor_VS_RIFClass2")
    main(RIF_donor_list=RIF_donor_list2, normal_donor_list=RIF_donor_list1, special_path_str="RIFClass1_VS_RIFClass2")
