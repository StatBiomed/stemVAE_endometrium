# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：utils_project.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/6/10 18:49 
"""
import logging

_logger = logging.getLogger(__name__)

from .GPU_manager_pytorch import auto_select_gpu_and_cpu, check_memory, check_gpu_memory

from torch.nn import Parameter
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import os
from stemVAE import *
from .utils_plot import *

import time
import random
import json
import anndata
import pandas as pd

import numpy as np
import scanpy as sc


def read_saved_temp_result_json(file_name):
    with open(file_name, "r") as json_file:
        data_list = []
        for line in json_file:
            json_obj = json.loads(line)
            data_list.append(json_obj)

    # 还原 DataFrame 对象
    restored_dic = {}
    for em_info in data_list:
        for embryo_id, predicted_df in em_info.items():
            # 将 JSON 数据转换为 DataFrame
            value2 = predicted_df.strip().split('\n')
            # 解析每行的 JSON 数据并构建 DataFrame 列表
            dataframes = []
            for line in value2:
                json_obj = json.loads(line)
                df = pd.DataFrame([json_obj])
                dataframes.append(df)

            # 合并 DataFrame 列表为一个 DataFrame
            final_df = pd.concat(dataframes, ignore_index=True)
            final_df.set_index("cell_index", inplace=True)
            restored_dic[embryo_id] = final_df
        # 显示还原后的 DataFrame

    return restored_dic


def str2bool(str):
    return True if str.lower() == 'true' else False


def setup_model(y, time, latent_dim, device):
    """ pro.infer.elbo
    Users are therefore strongly encouraged to use this interface in
        conjunction with ``pyro.settings.set(module_local_params=True)`` which
        will override the default implicit sharing of parameters across
        :class:`~pyro.nn.PyroModule` instances.
    """
    pyro.settings.set(module_local_params=True)
    # we setup the mean of our prior over X
    X_prior_mean = torch.zeros(y.size(1), latent_dim)  # shape: 437 x 2
    X_prior_mean[:, 0] = time
    _logger.info("Set latent variable x with prior info:{}".format(X_prior_mean))

    kernel = gp.kernels.RBF(input_dim=latent_dim, lengthscale=torch.ones(latent_dim)).to(device)

    # we clone here so that we don't change our prior during the course of training
    X = Parameter(X_prior_mean.clone()).to(device)
    y = y.to(device)
    # we will use SparseGPRegression model with num_inducing=32;
    # initial values for Xu are sampled randomly from X_prior_mean
    import pyro.ops.stats as stats
    Xu = stats.resample(X_prior_mean.clone(), 32).to(device)
    # jitter: A small positive term which is added into the diagonal part of a covariance matrix to help stablize its Cholesky decomposition.
    gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01).to(device), jitter=1e-4).to(device)
    # we use `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
    gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean.to(device), 0.1).to_event())
    gplvm.autoguide("X", dist.Normal)
    return gplvm


def predict_on_one_donor(gplvm, cell_time, sc_expression_train, sc_expression_test, golbal_path, file_path, donor,
                         sample_num=10, args=None):
    # ---------------------------------------------- Freeze gplvm for prediction  --------------------------------------------------
    """
    After training GPLVM, we get good hyperparameters for the data. 
    Then when new data is provided, we train GPLVM again with these fixed hyperparameters to learn X.
    """
    # Freeze gplvm
    # for param in gplvm.parameters():
    #     param.requires_grad_(False)
    pyro.settings.set(module_local_params=False)
    gplvm.mode = "guide"
    # sample 10 times
    sample_y_data = dict()
    labels = cell_time["time"].unique()
    sc_expression_df_copy = sc_expression_train.copy()
    for _cell_name in sc_expression_train.index.values:
        sc_expression_df_copy.rename(index={_cell_name: cell_time.loc[_cell_name]["time"]}, inplace=True)
    _logger.info("Sample from latent space {} times".format(sample_num))
    for i in range(sample_num):
        _X_one_sample = gplvm.X
        _Y_one_sample = gplvm.forward(_X_one_sample)
        # _Y_one_sample[0] is loc, _Y_one_sample[1] is var
        _Y_one_sample_df = pd.DataFrame(data=_Y_one_sample[0].cpu().detach().numpy().T,
                                        index=_X_one_sample[:, 0].cpu().detach().numpy(),
                                        columns=sc_expression_train.columns)
        for i, label in enumerate(labels):
            if label not in sample_y_data.keys():
                sample_y_data[label] = _Y_one_sample_df[sc_expression_df_copy.index == label]
            else:
                sample_y_data[label] = sample_y_data[label]._append(
                    _Y_one_sample_df[sc_expression_df_copy.index == label])

    each_type_num = np.inf
    for key in sample_y_data.keys():
        _logger.info("label {} have sampled {} cells".format(key, len(sample_y_data[key])))
        if len(sample_y_data[key]) < each_type_num:
            each_type_num = len(sample_y_data[key])
    for key in sample_y_data.keys():
        sample_y_data[key] = sample_y_data[key].sample(n=each_type_num)

    # # filter sc_expression_test; 2023-06-19 20:22:38 don't filter gene for test data, so delete code here
    # for _gene in sc_expression_test:
    #     if np.var(sc_expression_test[_gene].values) == 0:
    #         sc_expression_test = sc_expression_test.drop(_gene, axis=1)
    from utils_project import cosSim
    sample_y_data_df = pd.DataFrame(columns=sc_expression_train.columns)
    # sample_y_data_df_attr_time =pd.DataFrame(columns=sc_expression_train.columns)
    for key in sample_y_data.keys():
        sample_y_data_df = sample_y_data_df._append(sample_y_data[key])
        # temp=pd.DataFrame(data=sample_y_data[key].values,columns=sc_expression_train.columns,index=[key for i in range(len(sample_y_data[key]))])
        # sample_y_data_df_attr_time=sample_y_data_df_attr_time._append(temp)
    # plot the generated cells
    # plot_latent_dim_image(_X_one_sample.shape[1], labels, X, sample_y_data_df_attr_time, golbal_path, file_path, "time", reorder_labels=True, args=args,special_str="Sampled_")

    sample_y_data_df = sample_y_data_df[sc_expression_test.columns]
    result_test_pseudotime = pd.DataFrame(index=sc_expression_test.index, columns=["pseudotime"])
    for test_index, row_test in sc_expression_test.iterrows():
        _cell_samilars = []
        for time, row_sampled in sample_y_data_df.iterrows():
            _cell_samilars.append([time, cosSim(row_test, row_sampled)])
        _cell_samilars = np.array(_cell_samilars)
        _cell_samilars = _cell_samilars[_cell_samilars[:, 1].argsort()][::-1][:10, 0]
        _cell_samilars = np.mean(_cell_samilars)
        result_test_pseudotime.loc[test_index]["pseudotime"] = _cell_samilars
    result_test_pseudotime.to_csv('{}/{}/test_onOneDonor_{}.csv'.format(golbal_path, file_path, donor), sep="\t",
                                  index=True, header=True)
    _logger.info("Test donor: {}, pseudotime for each cell is {}".format(donor, result_test_pseudotime))
    return result_test_pseudotime


def preprocessData(golbal_path, file_name, KNN_smooth_type, cell_info_file, gene_list=None, min_cell_num=50,
                   min_gene_num=100):
    """
    2023-08-23 10:02:28 careful use, has update (add more function) to function preprocessData_and_dropout_some_donor_or_gene.
    :param golbal_path:
    :param file_name:
    :param KNN_smooth_type:
    :param cell_info_file:
    :param gene_list:
    :param min_cell_num:
    :param min_gene_num:
    :return:
    """
    adata = anndata.read_csv(golbal_path + file_name, delimiter='\t')
    if gene_list is not None:
        overlap_gene = list(set(adata.obs_names) & set(gene_list))
        adata = adata[overlap_gene].copy()

    adata = adata.T  # 基因和cell转置矩阵
    # 数据数目统计
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    _shape = adata.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata.shape
    _logger.info("Drop cells with less than {} gene expression, drop genes which none expression in {} samples".format(
        min_gene_num, min_cell_num))
    _logger.info("After filter, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _logger.info("Finish normalize per cell, so that every cell has the same total count after normalization.")

    denseM = adata.X
    # denseM = KNN_smoothing_start(denseM, type=KNN_smooth_type)
    # _logger.info("Finish smooth by {} method.".format(KNN_smooth_type))
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    _logger.info("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=adata.var.index, index=adata.obs.index)
    cell_time = pd.read_csv(golbal_path + cell_info_file, sep="\t", index_col=0)
    cell_time = cell_time.loc[sc_expression_df.index]
    _logger.info("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_time.shape))

    return sc_expression_df, cell_time


def downSample_matrix(matrix, target_location="row", reduce_multiple=10):
    matrix = matrix.astype("float64")
    import numpy.random as npr
    npr.seed(123)
    # Normalize matrix along rows
    if target_location == "row":
        row_sums = matrix.sum(axis=1)
        print(row_sums)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix)):
            if row_sums[i] == 0:
                row_sample = matrix[i]
            else:
                row_sample = np.random.multinomial(row_sums[i] / reduce_multiple, normalized_matrix[i])
            downsampled_matrix[i] = row_sample

        return downsampled_matrix
    elif target_location == "col":
        col_sums = matrix.sum(axis=0)
        print(col_sums)
        normalized_matrix = matrix / col_sums[np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix[0])):
            if col_sums[i] == 0:
                col_sample = matrix[:, i]
            else:
                col_sample = np.random.multinomial(col_sums[i] / reduce_multiple, normalized_matrix[:, i])
            downsampled_matrix[:, i] = col_sample

        return downsampled_matrix
    elif target_location == "line":
        # Flatten the matrix into a 1D array
        flattened_matrix = matrix.flatten().astype('float64')
        # Normalize the flattened array to create probabilities
        probabilities = flattened_matrix / np.sum(flattened_matrix)
        print(probabilities)
        # Generate multinomial samples
        print(flattened_matrix.sum())
        samples = np.random.multinomial(flattened_matrix.sum() / reduce_multiple, probabilities)

        # Reshape the samples back to the original matrix shape
        downsampled_matrix = samples.reshape(matrix.shape)

        return downsampled_matrix


def cosSim(x, y):
    '''
    余弦相似度
    '''
    tmp = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(tmp / float(non), 9)


def eculidDisSim(x, y):
    '''
    欧几里得相似度
    '''
    return np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattanDisSim(x, y):
    '''
    曼哈顿相似度
    '''
    return sum(abs(a - b) for a, b in zip(x, y))


def pearsonrSim(x, y):
    '''
    皮尔森相似度
    '''
    from scipy.stats import pearsonr
    return pearsonr(x, y)[0]


# def donor_resort_key(item):
#     import re
#     match = re.match(r'LH(\d+)_([0-9PGT]+)', item)
#     if match:
#         num1 = int(match.group(1))
#         num2 = match.group(2)
#         if num2 == 'PGT':
#             num2 = 'ZZZ'  # 将'LH7_PGT'替换为'LH7_ZZZ'，确保'LH7_PGT'在'LH7'后面
#         return num1, num2
#     return item
#
#
# def RIFdonor_resort_key(item):
#     import re
#     match = re.match(r'RIF_(\d+)', item)
#     if match:
#         num1 = int(match.group(1))
#         return num1
#     return item


def test_on_newDataset(sc_expression_train, data_golbal_path, result_save_path, KNN_smooth_type, runner, experiment,
                       config, latent_dim,
                       special_path_str, time_standard_type, test_data_path):
    """
    2023-07-13 14:39:38 dandan share a new dataset (download from public database, with epi and fibro, different platfrom: ct and 10X)
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict
    _logger.info("Test on new dataset.")

    gene_list = sc_expression_train.columns

    _logger.info("Test on dataset: {}".format(test_data_path))
    gene_dic = dict()
    file_name_test = test_data_path + "/data_count.csv"
    cell_info_file_test = test_data_path + "/cell_info.csv"
    sc_expression_df_test, cell_time_test = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, file_name_test,
                                                                                          KNN_smooth_type, cell_info_file_test,
                                                                                          gene_list=gene_list, min_cell_num=0,
                                                                                          min_gene_num=10)

    loss_gene = list(set(gene_list) - set(sc_expression_df_test.columns))
    _logger.info("loss {} gene in test data, set them to 0".format(len(loss_gene)))
    _logger.info("test data don't have gene: {}".format(loss_gene))
    gene_dic["model_gene"] = list(gene_list)
    gene_dic["loss_gene"] = list(loss_gene)
    gene_dic["testdata_gene"] = list(sc_expression_df_test.columns)
    for _g in loss_gene:
        sc_expression_df_test[_g] = 0
    x_sc_test = torch.tensor(sc_expression_df_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    train_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if test_clf_result.shape[1] == 1:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is continues.")
    else:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result = np.argmax(test_clf_result, axis=1)
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is discrete.")

    _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
    if not os.path.exists(_save_path):
        os.makedirs(_save_path)
    _save_file_name = "{}/{}_testOnExternal_{}.csv".format(_save_path, config['model_params']['name'],
                                                           test_data_path.replace("/", "").replace(".rds", "").replace(" ", "_"))

    test_clf_result_df.to_csv(_save_file_name, sep="\t")
    import json
    with open(_save_file_name.replace(".csv", "geneUsed.json"), 'w') as f:
        json.dump(gene_dic, f)
    _logger.info("result save at: {}".format(_save_file_name))


def test_on_newDonor(test_donor_name, sc_expression_test, runner, experiment, predict_donors_dic):
    """
    2023-07-16 15:05:18 dandan share 10 RIF donor, test on these donor
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict

    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    test_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
    if test_clf_result.shape[1] == 1:
        _logger.info("predicted time of test donor is continuous.")
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=np.squeeze(test_clf_result),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    test_latent_info_dic = {"mu": test_latent_mu_result,
                            "log_var": test_latent_log_var_result,
                            "label_true": np.zeros(len(test_latent_mu_result)),
                            "label_dic": {"test": "test"},
                            "donor_index": np.zeros(len(test_latent_mu_result)) - 1,
                            "donor_dic": {test_donor_name: -1}}
    return predict_donors_dic, test_clf_result, test_latent_info_dic


def one_fold_test(fold, donor_list, sc_expression_df, donor_dic, batch_dic,
                  special_path_str,
                  cell_time, time_standard_type, config, args,
                  device=None, plot_trainingLossLine=True, plot_latentSpaceUmap=True,
                  time_saved_asFloat=False, batch_size=None,
                  max_attempts=10000000):
    """
    use donor_list[fold] as test data, and use other donors as train data,
    then train a vae model with {latent dim} in latent space
    :param fold:
    :param donor_list:
    :param sc_expression_df:
    :param donor_dic:
    :param golbal_path:
    :param file_path:
    :param latent_dim:
    :param special_path_str:
    :param cell_time:
    :param time_standard_type:
    :param config:
    :param args:
    :param predict_donors_dic:
    :param device:
    :param batch_dim:
    :param plot_trainingLossLine:
    :param plot_latentSpaceUmap:
    :return:
    """
    from stemVAE.experiment import VAEXperiment
    from stemVAE.dataset import SupervisedVAEDataset
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict, SupervisedVAEDataset_onlyTrain
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pathlib import Path
    predict_donors_dic = dict()
    # ----------------------------------------split Train and Test dataset-----------------------------------------------------
    _logger.info("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))
    subFold_save_file_path = "{}{}/{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str,
                                               donor_list[fold])

    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time["donor"] != donor_list[fold]]]
    sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time["donor"] == donor_list[fold]]]

    # we need to transpose data to correct its shape
    x_sc_train = torch.tensor(sc_expression_train.values, dtype=torch.get_default_dtype()).t()
    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))

    # trans y_time

    if time_saved_asFloat:  # 2023-08-06 13:25:48 trans 8.50000 to 850; 9.2500000 to 925; easy to  manipulate.
        cell_time_dic = dict(zip(cell_time.index, cell_time["time"]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic) * 100).astype(int))
        y_time_test = x_sc_test.new_tensor(np.array(sc_expression_test.index.map(cell_time_dic) * 100).astype(int))

    else:
        y_time_train = x_sc_train.new_tensor(
            [int(cell_time.loc[_cell_name]["time"].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_train.index.values])
        y_time_test = x_sc_test.new_tensor(
            [int(cell_time.loc[_cell_name]["time"].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_test.index.values])
    donor_index_train = x_sc_train.new_tensor(
        [int(batch_dic[cell_time.loc[_cell_name]["donor"]]) for _cell_name in sc_expression_train.index.values])
    donor_index_test = x_sc_test.new_tensor(
        [int(batch_dic[cell_time.loc[_cell_name]["donor"]]) for _cell_name in sc_expression_test.index.values])

    # for classification model with discrete time cannot use sigmoid and logit time type
    y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type, capture_time_other=y_time_test)
    y_time_nor_test, label_dic = trans_time(y_time_test, time_standard_type, label_dic_train=label_dic)
    _logger.info("label dictionary: {}".format(label_dic))
    _logger.info("Normalize train y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_train), y_time_train.shape, np.unique(y_time_nor_train)))
    _logger.info("Normalize test y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_test), y_time_test.shape, np.unique(y_time_nor_test)))

    # ------------------------------------------- Set up VAE model and Start train process -------------------------------------------------
    _logger.info("Start training with epoch: {}. ".format(args.train_epoch_num))

    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    ## 打印模型的权重和偏置
    # for name, param in MyVAEModel.named_parameters():
    #     print(name, param)
    train_data = [[x_sc_train[:, i], y_time_nor_train[i], donor_index_train[i]] for i in range(x_sc_train.shape[1])]
    test_data = [[x_sc_test[:, i], y_time_nor_test[i], donor_index_test[i]] for i in range(x_sc_test.shape[1])]
    if batch_size is None:
        _logger.info("batch size is none, so don't set batch")
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=len(train_data), val_batch_size=len(test_data),
                                    test_batch_size=len(test_data), predict_batch_size=len(test_data),
                                    label_dic=label_dic)
    else:
        _logger.info("batch size is {}".format(batch_size))
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=batch_size, val_batch_size=batch_size,
                                    test_batch_size=batch_size, predict_batch_size=batch_size,
                                    label_dic=label_dic)
    # data.setup("train")
    experiment = VAEXperiment(MyVAEModel, config['exp_params'])

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # 2023-09-07 20:34:25 add check memory
    check_memory(max_attempts=max_attempts)
    device = auto_select_gpu_and_cpu(max_attempts=max_attempts)
    _logger.info("Auto select run on {}".format(device))

    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=args.train_epoch_num
                     )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, data)

    # test the model
    _logger.info("this epoch final, on test data:{}".format(runner.test(experiment, data)))
    # predict on train data
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    train_result = runner.predict(experiment, data_predict)
    train_clf_result, train_latent_mu_result, train_latent_log_var_result = train_result[0][0], train_result[0][1], \
        train_result[0][2]
    # predict on test data
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if (np.isnan(np.squeeze(test_clf_result)).any()):
        _logger.info("The Array contain NaN values")
    else:
        _logger.info("The Array does not contain NaN values")
    if test_clf_result.shape[1] == 1:
        # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
        _logger.info("predicted time of test donor is continuous.")
        try:
            predict_donors_dic[donor_list[fold]] = pd.DataFrame(data=np.squeeze(test_clf_result, axis=1),
                                                                index=sc_expression_test.index, columns=["pseudotime"])
        except:
            print("error here")
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donors_dic[donor_list[fold]] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                            index=sc_expression_test.index, columns=["pseudotime"])
    # acc = torch.tensor(torch.sum(labels_pred == y_time_nor_test).item() / (len(y_time_nor_test) * 1.0))
    # # ---------------------------------------------- plot sub result of training process for check  --------------------------------------------------
    if plot_trainingLossLine:
        _logger.info("Plot training loss line for check.")

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tags = EventAccumulator(tb_logger.log_dir).Reload().Tags()['scalars']
        _logger.info("All tags in logger: {}".format(tags))
        # Retrieve and print the metric results
        plot_tag_list = ["train_loss_epoch", "val_loss", "test_loss_epoch"]
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        plot_tag_list = ["train_clf_loss_epoch", "val_clf_loss", "test_clf_loss_epoch"]
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        plot_tag_list = ["train_Reconstruction_loss_epoch", "val_Reconstruction_loss", "test_Reconstruction_loss_epoch"]
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        try:
            plot_tag_list = ["train_batchEffect_loss_epoch", "val_batchEffect_loss", "test_batchEffect_loss_epoch"]
            plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        except:
            _logger.info("No batchEffect decoder.")
    else:
        _logger.info("Don't plot training loss line for check.")

    # # # ---------------------------------------------- plot sub latent space of sub model for check  --------------------------------------------------
    if plot_latentSpaceUmap:
        _logger.info("Plot each fold's UMAP of latent space for check.")
        umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_mu_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_mu_donor", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_logVar_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_logVar_donor", drop_batch_dim=0)
        try:
            umap_vae_latent_space(test_latent_mu_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_mu_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_mu_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_mu_donor", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_logVar_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_logVar_donor", drop_batch_dim=0)
        except:
            _logger.info("Too few test cells, can't plot umap of the latent space: {}.".format(len(y_time_nor_test)))
    else:
        _logger.info("Don't plot each fold's UMAP of latent space for check.")

    # ---------------------------------------------- save sub model parameters for check  --------------------------------------------------
    _logger.info(
        "encoder and decoder structure: {}".format({"encoder": MyVAEModel.encoder, "decoder": MyVAEModel.decoder}))
    _logger.info("clf-decoder: {}".format({"clf-decoder": MyVAEModel.clf_decoder}))
    torch.save(MyVAEModel, tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth')
    # _logger.info("detail information about structure save at： {}".format(tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth'))

    del MyVAEModel
    del runner
    del experiment
    # 清除CUDA缓存
    torch.cuda.empty_cache()
    return predict_donors_dic, test_clf_result, label_dic


def process_fold(fold, donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time,
                 time_standard_type, config, args, batch_size, adversarial_train_bool=False):
    # _logger = logging.getLogger(__name__)
    time.sleep(random.randint(10, 100))
    _logger.info("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))

    try:
        predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, donor_list, sc_expression_df, donor_dic,
                                                                      batch_dic,
                                                                      special_path_str,
                                                                      cell_time, time_standard_type, config, args,
                                                                      plot_trainingLossLine=False, plot_latentSpaceUmap=False,
                                                                      time_saved_asFloat=True,
                                                                      batch_size=batch_size,
                                                                      max_attempts=5)
        temp_save_dic(special_path_str, config, time_standard_type, predict_donor_dic.copy(), label_dic=label_dic)
        result = (predict_donor_dic, test_clf_result, label_dic)
        return result
    except Exception as e:
        # 如果有异常，记录错误日志，日志消息将通过队列传递给主进程
        _logger.error("Note!!! Error {} processing data at embryo {}".format(donor_list[fold], e))
        return None


def temp_save_dic(special_path_str, config, time_standard_type, predict_donor_dic, label_dic=None):
    import fcntl
    temp_save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}_temp.json".format(
        _logger.root.handlers[0].baseFilename.replace(".log", ""),
        special_path_str,
        config['model_params']['name'],
        time_standard_type)
    for key, value in predict_donor_dic.items():
        value.index.name = "cell_index"
        _dic = value.reset_index().to_json(orient='records', lines=True)
        predict_donor_dic.update({key: _dic})
    # 检查 JSON 文件是否被占用，如果被占用则等待60秒

    while True:
        try:
            with open(temp_save_file_name, "a") as json_file:
                # 尝试获取文件锁
                fcntl.flock(json_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 文件没有被占用，写入结果并释放文件锁
                json.dump(predict_donor_dic, json_file)
                json_file.write("\n")  # 添加换行符以分隔每个结果
                fcntl.flock(json_file.fileno(), fcntl.LOCK_UN)
                break
        except BlockingIOError:
            # 文件被占用，等待60秒
            time.sleep(60)
    _logger.info("Temp predict_donor_dic save at {}".format(temp_save_file_name))
    if label_dic is not None:
        for key, value in label_dic.items():
            label_dic[key] = float(value)
        while True:
            try:
                with open(temp_save_file_name.replace("temp.json", "labelDic.json"), "w") as json_file:
                    # 尝试获取文件锁
                    fcntl.flock(json_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 文件没有被占用，写入结果并释放文件锁
                    json.dump(label_dic, json_file)
                    json_file.write("\n")  # 添加换行符以分隔每个结果
                    fcntl.flock(json_file.fileno(), fcntl.LOCK_UN)
                    break
            except BlockingIOError:
                # 文件被占用，等待60秒
                time.sleep(60)
        _logger.info("Save label dic at {}".format(temp_save_file_name.replace("temp.json", "labelDic.json")))


def str2bool(str):
    return True if str.lower() == 'true' else False


def setup_model(y, time, latent_dim, device):
    """ pro.infer.elbo
    Users are therefore strongly encouraged to use this interface in
        conjunction with ``pyro.settings.set(module_local_params=True)`` which
        will override the default implicit sharing of parameters across
        :class:`~pyro.nn.PyroModule` instances.
    """
    pyro.settings.set(module_local_params=True)
    # we setup the mean of our prior over X
    X_prior_mean = torch.zeros(y.size(1), latent_dim)  # shape: 437 x 2
    X_prior_mean[:, 0] = time
    _logger.info("Set latent variable x with prior info:{}".format(X_prior_mean))

    kernel = gp.kernels.RBF(input_dim=latent_dim, lengthscale=torch.ones(latent_dim)).to(device)

    # we clone here so that we don't change our prior during the course of training
    X = Parameter(X_prior_mean.clone()).to(device)
    y = y.to(device)
    # we will use SparseGPRegression model with num_inducing=32;
    # initial values for Xu are sampled randomly from X_prior_mean
    import pyro.ops.stats as stats
    Xu = stats.resample(X_prior_mean.clone(), 32).to(device)
    # jitter: A small positive term which is added into the diagonal part of a covariance matrix to help stablize its Cholesky decomposition.
    gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01).to(device), jitter=1e-4).to(device)
    # we use `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
    gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean.to(device), 0.1).to_event())
    gplvm.autoguide("X", dist.Normal)
    return gplvm


def predict_on_one_donor(gplvm, cell_time, sc_expression_train, sc_expression_test, golbal_path, file_path, donor,
                         sample_num=10, args=None):
    # ---------------------------------------------- Freeze gplvm for prediction  --------------------------------------------------
    import matplotlib.pyplot as plt
    """
    After training GPLVM, we get good hyperparameters for the data. 
    Then when new data is provided, we train GPLVM again with these fixed hyperparameters to learn X.
    """
    # Freeze gplvm
    # for param in gplvm.parameters():
    #     param.requires_grad_(False)
    pyro.settings.set(module_local_params=False)
    gplvm.mode = "guide"
    # sample 10 times
    sample_y_data = dict()
    labels = cell_time["time"].unique()
    sc_expression_df_copy = sc_expression_train.copy()
    for _cell_name in sc_expression_train.index.values:
        sc_expression_df_copy.rename(index={_cell_name: cell_time.loc[_cell_name]["time"]}, inplace=True)
    _logger.info("Sample from latent space {} times".format(sample_num))
    for i in range(sample_num):
        _X_one_sample = gplvm.X
        _Y_one_sample = gplvm.forward(_X_one_sample)
        # _Y_one_sample[0] is loc, _Y_one_sample[1] is var
        _Y_one_sample_df = pd.DataFrame(data=_Y_one_sample[0].cpu().detach().numpy().T,
                                        index=_X_one_sample[:, 0].cpu().detach().numpy(),
                                        columns=sc_expression_train.columns)
        for i, label in enumerate(labels):
            if label not in sample_y_data.keys():
                sample_y_data[label] = _Y_one_sample_df[sc_expression_df_copy.index == label]
            else:
                sample_y_data[label] = sample_y_data[label]._append(
                    _Y_one_sample_df[sc_expression_df_copy.index == label])

    each_type_num = np.inf
    for key in sample_y_data.keys():
        _logger.info("label {} have sampled {} cells".format(key, len(sample_y_data[key])))
        if len(sample_y_data[key]) < each_type_num:
            each_type_num = len(sample_y_data[key])
    for key in sample_y_data.keys():
        sample_y_data[key] = sample_y_data[key].sample(n=each_type_num)

    # # filter sc_expression_test; 2023-06-19 20:22:38 don't filter gene for test data, so delete code here
    # for _gene in sc_expression_test:
    #     if np.var(sc_expression_test[_gene].values) == 0:
    #         sc_expression_test = sc_expression_test.drop(_gene, axis=1)
    from utils_project import cosSim
    sample_y_data_df = pd.DataFrame(columns=sc_expression_train.columns)
    # sample_y_data_df_attr_time =pd.DataFrame(columns=sc_expression_train.columns)
    for key in sample_y_data.keys():
        sample_y_data_df = sample_y_data_df._append(sample_y_data[key])
        # temp=pd.DataFrame(data=sample_y_data[key].values,columns=sc_expression_train.columns,index=[key for i in range(len(sample_y_data[key]))])
        # sample_y_data_df_attr_time=sample_y_data_df_attr_time._append(temp)
    # plot the generated cells
    # plot_latent_dim_image(_X_one_sample.shape[1], labels, X, sample_y_data_df_attr_time, golbal_path, file_path, "time", reorder_labels=True, args=args,special_str="Sampled_")

    sample_y_data_df = sample_y_data_df[sc_expression_test.columns]
    result_test_pseudotime = pd.DataFrame(index=sc_expression_test.index, columns=["pseudotime"])
    for test_index, row_test in sc_expression_test.iterrows():
        _cell_samilars = []
        for time, row_sampled in sample_y_data_df.iterrows():
            _cell_samilars.append([time, cosSim(row_test, row_sampled)])
        _cell_samilars = np.array(_cell_samilars)
        _cell_samilars = _cell_samilars[_cell_samilars[:, 1].argsort()][::-1][:10, 0]
        _cell_samilars = np.mean(_cell_samilars)
        result_test_pseudotime.loc[test_index]["pseudotime"] = _cell_samilars
    result_test_pseudotime.to_csv('{}/{}/test_onOneDonor_{}.csv'.format(golbal_path, file_path, donor), sep="\t",
                                  index=True, header=True)
    _logger.info("Test donor: {}, pseudotime for each cell is {}".format(donor, result_test_pseudotime))
    return result_test_pseudotime


def preprocessData(golbal_path, file_name, KNN_smooth_type, cell_info_file, gene_list=None, min_cell_num=50,
                   min_gene_num=100):
    """
    2023-08-23 10:02:28 careful use, has update (add more function) to function preprocessData_and_dropout_some_donor_or_gene.
    :param golbal_path:
    :param file_name:
    :param KNN_smooth_type:
    :param cell_info_file:
    :param gene_list:
    :param min_cell_num:
    :param min_gene_num:
    :return:
    """
    adata = anndata.read_csv(golbal_path + file_name, delimiter='\t')
    if gene_list is not None:
        overlap_gene = list(set(adata.obs_names) & set(gene_list))
        adata = adata[overlap_gene].copy()

    adata = adata.T  # 基因和cell转置矩阵
    # 数据数目统计
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    _shape = adata.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata.shape
    _logger.info("Drop cells with less than {} gene expression, drop genes which none expression in {} samples".format(
        min_gene_num, min_cell_num))
    _logger.info("After filter, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _logger.info("Finish normalize per cell, so that every cell has the same total count after normalization.")

    denseM = adata.X
    # denseM = KNN_smoothing_start(denseM, type=KNN_smooth_type)
    # _logger.info("Finish smooth by {} method.".format(KNN_smooth_type))
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    _logger.info("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=adata.var.index, index=adata.obs.index)
    cell_time = pd.read_csv(golbal_path + cell_info_file, sep="\t", index_col=0)
    cell_time = cell_time.loc[sc_expression_df.index]
    _logger.info("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_time.shape))

    return sc_expression_df, cell_time


def downSample_matrix(matrix, target_location="row", reduce_multiple=10):
    matrix = matrix.astype("float64")
    import numpy.random as npr
    npr.seed(123)
    # Normalize matrix along rows
    if target_location == "row":
        row_sums = matrix.sum(axis=1)
        print(row_sums)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix)):
            if row_sums[i] == 0:
                row_sample = matrix[i]
            else:
                row_sample = np.random.multinomial(row_sums[i] / reduce_multiple, normalized_matrix[i])
            downsampled_matrix[i] = row_sample

        return downsampled_matrix
    elif target_location == "col":
        col_sums = matrix.sum(axis=0)
        print(col_sums)
        normalized_matrix = matrix / col_sums[np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix[0])):
            if col_sums[i] == 0:
                col_sample = matrix[:, i]
            else:
                col_sample = np.random.multinomial(col_sums[i] / reduce_multiple, normalized_matrix[:, i])
            downsampled_matrix[:, i] = col_sample

        return downsampled_matrix
    elif target_location == "line":
        # Flatten the matrix into a 1D array
        flattened_matrix = matrix.flatten().astype('float64')
        # Normalize the flattened array to create probabilities
        probabilities = flattened_matrix / np.sum(flattened_matrix)
        print(probabilities)
        # Generate multinomial samples
        print(flattened_matrix.sum())
        samples = np.random.multinomial(flattened_matrix.sum() / reduce_multiple, probabilities)

        # Reshape the samples back to the original matrix shape
        downsampled_matrix = samples.reshape(matrix.shape)

        return downsampled_matrix


def preprocessData_and_dropout_some_donor_or_gene(golbal_path, file_name, KNN_smooth_type, cell_info_file,
                                                  drop_out_donor=None, donor_attr="donor", gene_list=None,
                                                  drop_out_cell_type=None,
                                                  min_cell_num=50, min_gene_num=100, keep_sub_type_with_cell_num=None,
keep_sub_donor_with_cell_num=None,
                                                  external_file_name=None, external_cell_info_file=None,
                                                  # external_cellId_list=None,
                                                  downSample_on_testData_bool=False, test_donor=None,
                                                  downSample_location_type=None,
                                                  augmentation_on_trainData_bool=False,
                                                  plot_boxPlot_bool=False,
                                                  special_path_str="",
                                                  random_drop_cell_bool=False,
                                                  normalized_cellTotalCount=1e6):
    # drop should before perprocess sc data
    from .utils_plot import plot_boxPlot_nonExpGene_percentage_whilePreprocess
    _logger.info("the original sc expression anndata should be gene as row, cell as column")
    try:
        adata = anndata.read_csv("{}/{}".format(golbal_path, file_name), delimiter='\t')
    except:
        adata = anndata.read_csv("{}/{}".format(golbal_path, file_name), delimiter=',')
    _logger.info("read the original sc expression anndata with shape (gene, cell): {}".format(adata.shape))

    if external_file_name is not None:
        try:
            adata2 = anndata.read_csv("{}/{}".format(golbal_path, external_file_name), delimiter='\t')
        except:
            adata2 = anndata.read_csv("{}/{}".format(golbal_path, external_file_name), delimiter=',')
        _logger.info("read the external test dataset sc expression anndata with shape (gene, cell): {}".format(adata2.shape))
        # if external_cellId_list is not None:
        #     _logger.info(f"external data select cells by external_cellId_list with {len(external_cellId_list)} cells")
        #     adata2 = adata2[:, external_cellId_list].copy()
        adata2 = adata2.T.copy()
        sc.pp.filter_cells(adata2, min_genes=min_gene_num)
        adata2 = adata2.T.copy()
        # 2023-08-03 18:41:20 concat adata and adata2
        # 查找adata1和adata2中duplicate columns, that is the duplicate cell name
        duplicate_columns = set(adata.var_names) & set(adata2.var_names)
        # 删除adata2中的重复列
        adata2 = adata2[:, ~adata2.var_names.isin(duplicate_columns)]
        _logger.info("drop out {} duplicate cell (with the same cell name) from external data".format(len(duplicate_columns)))

        adata = anndata.concat([adata.copy(), adata2.copy()], axis=1)
        _logger.info("merged sc data and external test dataset with shape (gene, cell): {}".format(adata.shape))

    if gene_list is not None:
        overlap_gene = list(set(adata.obs_names) & set(gene_list))
        adata = adata[overlap_gene].copy()
        _logger.info("with gene list require, adata filted with {} genes.".format(len(overlap_gene)))
    adata = adata.T  # 基因和cell转置矩阵
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    cell_time = pd.read_csv(golbal_path + cell_info_file, sep="\t", index_col=0)

    if external_cell_info_file is not None:
        external_cell_time = pd.read_csv(golbal_path + external_cell_info_file, sep="\t", index_col=0)
        _logger.info("Import external cell info dataframe with (cell, attr-num): {}".format(external_cell_time.shape))

        external_cell_time = external_cell_time.drop(duplicate_columns, axis=0)
        _logger.info("drop out {} duplicate cell(with the same cell name)".format(len(duplicate_columns)))

        cell_time = pd.concat([cell_time, external_cell_time])
        _logger.info("merged sc cell info and external cell info dataframe with (cell, attr-num): {}".format(cell_time.shape))

    if drop_out_donor is not None:
        drop_out_cell = list(
            set(cell_time.loc[cell_time[donor_attr].isin(list(drop_out_donor))].index) & set(adata.obs.index))
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()
        _logger.info("After drop out {}, get expression dataframe with shape (cell, gene): {}.".format(drop_out_donor,
                                                                                                       adata.shape))
    if drop_out_cell_type is not None:
        _logger.info(f"drop out {len(drop_out_cell_type)} cell type: {drop_out_cell_type}")
        # 2023-11-07 11:05:26 for Joy project, she focus on epi cells, so remove fibro cells, and the attr name in cell_time is "broad_celltype"
        drop_out_cell = list(set(cell_time.loc[cell_time["broad_celltype"].isin(list(drop_out_cell_type))].index) & set(adata.obs.index))
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()
        _logger.info("After drop out {}, get expression dataframe with shape (cell, gene): {}.".format(drop_out_donor, adata.shape))
    if plot_boxPlot_bool:
        try:
            plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                               special_path_str, test_donor,
                                                               special_file_str="1InitialImportData")
            plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                              special_path_str, test_donor,
                                                              special_file_str="1InitialImportData")
        except:
            print("some error while plot before boxplot.")
    if random_drop_cell_bool:
        _logger.info("To fast check model performance, random dropout cells, only save few cells to train and test")
        random.seed(123)
        # 随机生成不重复的样本索引
        random_indices = random.sample(range(adata.shape[0]), int(adata.n_obs / 10), )

        # 从 anndata 对象中获取选定的样本数据
        adata = adata[random_indices, :].copy()
        _logger.info("After random select 1/10 samples from adata, remain adata shape: {}".format(adata.shape))
        if plot_boxPlot_bool:
            try:
                plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                                   special_path_str, test_donor,
                                                                   special_file_str="2RandomDropPartCell")
                plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                                  special_path_str, test_donor,
                                                                  special_file_str="2RandomDropPartCell")
            except:
                print("some error while plot before boxplot.")

    if downSample_on_testData_bool:
        _logger.info("Down sample on test donor: {}, downSample location type: {}".format(test_donor, downSample_location_type))
        test_cell = list(set(cell_time.loc[cell_time[donor_attr] == test_donor].index) & set(adata.obs.index))
        adata_test = adata[test_cell].copy()

        test_dataframe = pd.DataFrame(data=adata_test.X, columns=adata_test.var.index, index=adata_test.obs.index)
        temp = downSample_matrix(np.array(test_dataframe.values), target_location=downSample_location_type)
        downSample_test_dataframe = pd.DataFrame(data=temp, columns=test_dataframe.columns, index=test_dataframe.index)
        downSample_test_anndata = anndata.AnnData(downSample_test_dataframe)
        adata = adata[adata.obs_names.drop(test_cell)].copy()
        adata = anndata.concat([adata.copy(), downSample_test_anndata.copy()], axis=0)
        # downSample_test_dataframe.values.sum()

        if plot_boxPlot_bool:
            try:
                plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                                   special_path_str, test_donor,
                                                                   special_file_str=f"3DownSampleOnTestBy{downSample_location_type}")
                plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                                  special_path_str, test_donor,
                                                                  special_file_str=f"3DownSampleOnTestBy{downSample_location_type}")
            except:
                print("some error while plot before boxplot.")
    if augmentation_on_trainData_bool:
        _logger.info("Data augmentation on train set by down sample 1/10 and 1/3, downSample location type line")
        train_cell = list(set(cell_time.loc[cell_time[donor_attr] != test_donor].index) & set(adata.obs.index))
        _logger.info(f"the train cell number is {len(train_cell)}")
        adata_train = adata[train_cell].copy()

        train_dataframe = pd.DataFrame(data=adata_train.X, columns=adata_train.var.index, index=adata_train.obs.index)
        temp10 = downSample_matrix(np.array(train_dataframe.values), target_location="line", reduce_multiple=10)
        temp3 = downSample_matrix(np.array(train_dataframe.values), target_location="line", reduce_multiple=3)
        downSample10_train_df = pd.DataFrame(data=temp10, columns=train_dataframe.columns,
                                             index=train_dataframe.index + "_downSample10")
        downSample3_train_df = pd.DataFrame(data=temp3, columns=train_dataframe.columns,
                                            index=train_dataframe.index + "_downSample3")
        downSample10_train_anndata = anndata.AnnData(downSample10_train_df)
        downSample3_train_anndata = anndata.AnnData(downSample3_train_df)

        adata = anndata.concat([adata.copy(), downSample10_train_anndata.copy()], axis=0)
        adata = anndata.concat([adata.copy(), downSample3_train_anndata.copy()], axis=0)
        _logger.info(f"After concat downSample 1/10 and 1/3 train cell, the adata with {adata.n_obs} cell, {adata.n_vars} gene.")
        train_cell_time = cell_time.loc[train_cell]
        downSample10_train_cell_time = train_cell_time.copy()
        downSample10_train_cell_time.index = downSample10_train_cell_time.index + "_downSample10"
        downSample3_train_cell_time = train_cell_time.copy()
        downSample3_train_cell_time.index = downSample3_train_cell_time.index + "_downSample3"

        cell_time = pd.concat([cell_time, downSample10_train_cell_time])
        cell_time = pd.concat([cell_time, downSample3_train_cell_time])
        _logger.info(
            f"Also add the downSample cell info to cell time dataframe and shape to {cell_time.shape}, columns is {cell_time.columns}")
        if plot_boxPlot_bool:
            try:
                plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                                   special_path_str, test_donor,
                                                                   special_file_str="4DataAugmentationOnTrainByline")
                plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                                  special_path_str, test_donor,
                                                                  special_file_str="4DataAugmentationOnTrainByline")
            except:
                print("some error while plot before boxplot.")

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

    if plot_boxPlot_bool:
        try:
            plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                               special_path_str, test_donor,
                                                               special_file_str=f"5filterGene{min_cell_num}Cell{min_gene_num}")
            plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                              special_path_str, test_donor,
                                                              special_file_str=f"5filterGene{min_cell_num}Cell{min_gene_num}")
        except:
            print("some error while plot before boxplot.")
    if keep_sub_donor_with_cell_num is not None:
        # 2024-04-18 11:28:33 add for dandan NC review, test subset same cell number for each donor
        print(f"keep subset same cell number for each donor: {keep_sub_donor_with_cell_num}")
        drop_out_cell = []
        _cell_time = cell_time.loc[adata.obs_names]
        from collections import Counter
        _num_dic = dict(Counter(_cell_time["donor"]))
        _logger.info("In this dataset (donor, number of cells): {}".format(_num_dic))
        for _donor, _cell_num in _num_dic.items():
            if _cell_num > keep_sub_donor_with_cell_num:
                _cell_donor_df = _cell_time.loc[_cell_time["donor"] == _donor]
                _drop_cell = _cell_donor_df.sample(n=len(_cell_donor_df) - keep_sub_donor_with_cell_num,
                                                  random_state=0).index
                drop_out_cell += list(_drop_cell)
                _logger.info(f"Drop out {len(_drop_cell)} cells for donor { _donor}, from {_cell_num} to {keep_sub_donor_with_cell_num} cells")
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()
        _logger.info("After drop out {}, get expression dataframe with shape (cell, gene): {}.".format(drop_out_donor,
                                                                                                       adata.shape))

    if keep_sub_type_with_cell_num is not None:
        drop_out_cell = []
        _logger.info(
            "Random select {} cells in each sub celltype; if one sub type number less than the threshold, keep all.".format(
                keep_sub_type_with_cell_num))
        _cell_time = cell_time.loc[adata.obs_names]
        from collections import Counter
        _num_dic = dict(Counter(_cell_time["major_cell_type"]))
        _logger.info("In this cell type (sub cell type, number of cells): {}".format(_num_dic))
        for _cell_type, _cell_num in _num_dic.items():
            if _cell_num > keep_sub_type_with_cell_num:
                _cell_type_df = _cell_time.loc[_cell_time["major_cell_type"] == _cell_type]
                _drop_cell = _cell_type_df.sample(n=len(_cell_type_df) - keep_sub_type_with_cell_num,
                                                  random_state=0).index
                drop_out_cell += list(_drop_cell)
                _logger.info("Drop out {} cells for {}, which has {} cells before".format(len(_drop_cell), _cell_type,
                                                                                          _cell_num))
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()
        _logger.info("After drop out {}, get expression dataframe with shape (cell, gene): {}.".format(drop_out_donor,
                                                                                                       adata.shape))

    _logger.info("Drop cells with less than {} gene expression, drop genes which none expression in {} samples".format(
        min_gene_num, min_cell_num))
    _logger.info("After filter, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    gene_raw_total_count = pd.DataFrame(data=adata.X.sum(axis=0), index=adata.var_names, columns=["raw_total_count"])

    sc.pp.normalize_total(adata, target_sum=normalized_cellTotalCount)
    sc.pp.log1p(adata)
    _logger.info(f"Finish normalize per cell to {normalized_cellTotalCount}, "
                 f"so that every cell has the same total count after normalization.")
    if plot_boxPlot_bool:
        try:
            plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                               special_path_str, test_donor,
                                                               special_file_str=f"6NormalizeTo1e6AndLog")
            plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                              special_path_str, test_donor,
                                                              special_file_str=f"6NormalizeTo1e6AndLog")
        except:
            print("some error while plot before boxplot.")

    sc_expression_df = pd.DataFrame(data=adata.X, columns=adata.var.index, index=adata.obs.index)
    # 2023-08-05 15:12:28 for debug
    # sc_expression_df = sc_expression_df.sample(n=6000, random_state=0)

    denseM = sc_expression_df.values
    # denseM = KNN_smoothing_start(denseM, type=KNN_smooth_type)
    # _logger.info("Finish smooth by {} method.".format(KNN_smooth_type))
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    _logger.info("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=sc_expression_df.columns, index=sc_expression_df.index)

    cell_time = cell_time.loc[sc_expression_df.index]
    _logger.info("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_time.shape))
    _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
    if not os.path.exists(_save_path):
        os.makedirs(_save_path)
    cell_time.to_csv(f"{_save_path}/preprocessed_cell_info.csv")
    # gene_list = list(sc_expression_df.columns)
    # np.savetxt(f"{_save_path}/preprocessed_gene_info.csv", gene_list, delimiter="\t", encoding='utf-8', fmt="%s")

    gene_raw_total_count.to_csv(f"{_save_path}/preprocessed_gene_info.csv")
    return sc_expression_df, cell_time


def cosSim(x, y):
    '''
    余弦相似度
    '''
    tmp = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(tmp / float(non), 9)


def eculidDisSim(x, y):
    '''
    欧几里得相似度
    '''
    return np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattanDisSim(x, y):
    '''
    曼哈顿相似度
    '''
    return sum(abs(a - b) for a, b in zip(x, y))


def pearsonrSim(x, y):
    '''
    皮尔森相似度
    '''
    from scipy.stats import pearsonr
    return pearsonr(x, y)[0]


def trans_time(capture_time, time_standard_type, capture_time_other=None, label_dic_train=None):
    if label_dic_train is not None:  # the label dic is given by label_dic_train
        label_dic = label_dic_train
    elif time_standard_type == "log2":
        label_dic = {3: 0, 5: 0.528, 7: 0.774, 9: 0.936, 11: 1.057}
        # time = (capture_time - 2).log2() / 3
    elif time_standard_type == "0to1":
        label_dic = {3: 0, 5: 0.25, 7: 0.5, 9: 0.75, 11: 1}
        # time = (capture_time - min(capture_time)) / (max(capture_time) - min(capture_time))
    elif time_standard_type == "neg1to1":
        label_dic = {3: -1, 5: -0.5, 7: 0, 9: 0.5, 11: 1}
        # time = (capture_time - np.unique(capture_time).mean()) / (max(capture_time) - min(capture_time))
    elif time_standard_type == "labeldic":
        label_dic = {3: 0, 5: 1, 7: 2, 9: 3, 11: 4}
        # time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    elif time_standard_type == "sigmoid":
        label_dic = {3: -5, 5: -3.5, 7: 0, 9: 3.5, 11: 5}
        # time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    elif time_standard_type == "logit":
        label_dic = {3: -5, 5: -1.5, 7: 0, 9: 1.5, 11: 5}
        # time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    # if data_set=="test":
    elif time_standard_type == "acinardic":
        label_dic = {1: 0, 5: 1, 6: 2, 21: 3, 22: 4, 38: 5, 44: 6, 54: 7}
    elif time_standard_type[:6] == "embryo":
        multiple = int(time_standard_type.split("to")[-1])
        new_max = multiple
        new_min = -1 * new_max
        unique_time = np.unique(capture_time)
        if capture_time_other is not None:
            unique_time = np.concatenate((unique_time, np.unique(capture_time_other)))  # all time point
        # 计算最小值和最大值
        min_val = np.min(unique_time)
        max_val = np.max(unique_time)

        # 最小-最大归一化
        normalized_data = (unique_time - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
        # normalized_data = (unique_time - min_val) / (max_val - min_val) * 2 - 1

        # 构建字典，保留小数点后三位
        label_dic = {int(key): round(value, 3) for key, value in zip(unique_time, normalized_data)}
    elif time_standard_type == "organdic":  # 2023-11-02 10:50:09 add for Joy organ project
        # label_dic = {100: 1, 0: 0} # 2023-11-07 12:18:52 for "mesen" attr, don't use.
        label_dic = {i: i for i in np.unique(capture_time)}
    time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    return time, label_dic




def test_on_newDataset(sc_expression_train, data_golbal_path, result_save_path, KNN_smooth_type, runner, experiment,
                       config, latent_dim,
                       special_path_str, time_standard_type, test_data_path):
    """
    2023-07-13 14:39:38 dandan share a new dataset (download from public database, with epi and fibro, different platfrom: ct and 10X)
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict
    _logger.info("Test on new dataset.")

    gene_list = sc_expression_train.columns

    _logger.info("Test on dataset: {}".format(test_data_path))
    gene_dic = dict()
    file_name_test = test_data_path + "/data_count.csv"
    cell_info_file_test = test_data_path + "/cell_info.csv"
    sc_expression_df_test, cell_time_test = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, file_name_test,
                                                                                          KNN_smooth_type, cell_info_file_test,
                                                                                          gene_list=gene_list, min_cell_num=0,
                                                                                          min_gene_num=10)

    loss_gene = list(set(gene_list) - set(sc_expression_df_test.columns))
    _logger.info("loss {} gene in test data, set them to 0".format(len(loss_gene)))
    _logger.info("test data don't have gene: {}".format(loss_gene))
    gene_dic["model_gene"] = list(gene_list)
    gene_dic["loss_gene"] = list(loss_gene)
    gene_dic["testdata_gene"] = list(sc_expression_df_test.columns)
    for _g in loss_gene:
        sc_expression_df_test[_g] = 0
    x_sc_test = torch.tensor(sc_expression_df_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    train_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if test_clf_result.shape[1] == 1:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is continues.")
    else:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result = np.argmax(test_clf_result, axis=1)
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is discrete.")

    _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
    if not os.path.exists(_save_path):
        os.makedirs(_save_path)
    _save_file_name = "{}/{}_testOnExternal_{}.csv".format(_save_path, config['model_params']['name'],
                                                           test_data_path.replace("/", "").replace(".rds", "").replace(" ", "_"))

    test_clf_result_df.to_csv(_save_file_name, sep="\t")
    import json
    with open(_save_file_name.replace(".csv", "geneUsed.json"), 'w') as f:
        json.dump(gene_dic, f)
    _logger.info("result save at: {}".format(_save_file_name))


def test_on_newDonor(test_donor_name, sc_expression_test, runner, experiment, predict_donors_dic):
    """
    2023-07-16 15:05:18 dandan share 10 RIF donor, test on these donor
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict

    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    test_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
    if test_clf_result.shape[1] == 1:
        _logger.info("predicted time of test donor is continuous.")
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=np.squeeze(test_clf_result),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    test_latent_info_dic = {"mu": test_latent_mu_result,
                            "log_var": test_latent_log_var_result,
                            "label_true": np.zeros(len(test_latent_mu_result)),
                            "label_dic": {"test": "test"},
                            "donor_index": np.zeros(len(test_latent_mu_result)) - 1,
                            "donor_dic": {test_donor_name: -1}}
    return predict_donors_dic, test_clf_result, test_latent_info_dic


def one_fold_test(fold, donor_list, sc_expression_df, donor_dic, batch_dic,
                  special_path_str,
                  cell_time, time_standard_type, config, args,
                  donor_str="donor", time_str="time",
                  device=None, plot_trainingLossLine=True, plot_latentSpaceUmap=True,
                  time_saved_asFloat=False, batch_size=None,
                  max_attempts=10000000):
    """
    use donor_list[fold] as test data, and use other donors as train data,
    then train a vae model with {latent dim} in latent space
    :param fold:
    :param donor_list:
    :param sc_expression_df:
    :param donor_dic:
    :param golbal_path:
    :param file_path:
    :param latent_dim:
    :param special_path_str:
    :param cell_time:
    :param time_standard_type:
    :param config:
    :param args:
    :param predict_donors_dic:
    :param device:
    :param batch_dim:
    :param plot_trainingLossLine:
    :param plot_latentSpaceUmap:
    :return:
    """
    from stemVAE.experiment import VAEXperiment
    from stemVAE.dataset import SupervisedVAEDataset
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict, SupervisedVAEDataset_onlyTrain
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pathlib import Path
    predict_donors_dic = dict()
    # ----------------------------------------split Train and Test dataset-----------------------------------------------------
    _logger.info("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))
    subFold_save_file_path = "{}{}/{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str,
                                               donor_list[fold])

    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time[donor_str] != donor_list[fold]]]
    sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time[donor_str] == donor_list[fold]]]

    # we need to transpose data to correct its shape
    x_sc_train = torch.tensor(sc_expression_train.values, dtype=torch.get_default_dtype()).t()
    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))

    # trans y_time

    if time_saved_asFloat:  # 2023-08-06 13:25:48 trans 8.50000 to 850; 9.2500000 to 925; easy to  manipulate.
        cell_time_dic = dict(zip(cell_time.index, cell_time[time_str]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic) * 100).astype(int))
        y_time_test = x_sc_test.new_tensor(np.array(sc_expression_test.index.map(cell_time_dic) * 100).astype(int))

    else:
        y_time_train = x_sc_train.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_train.index.values])
        y_time_test = x_sc_test.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_test.index.values])
    donor_index_train = x_sc_train.new_tensor(
        [int(batch_dic[cell_time.loc[_cell_name][donor_str]]) for _cell_name in sc_expression_train.index.values])
    donor_index_test = x_sc_test.new_tensor(
        [int(batch_dic[cell_time.loc[_cell_name][donor_str]]) for _cell_name in sc_expression_test.index.values])

    # for classification model with discrete time cannot use sigmoid and logit time type
    y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type, capture_time_other=y_time_test)
    y_time_nor_test, label_dic = trans_time(y_time_test, time_standard_type, label_dic_train=label_dic)
    _logger.info("label dictionary: {}".format(label_dic))
    _logger.info("Normalize train y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_train), y_time_train.shape, np.unique(y_time_nor_train)))
    _logger.info("Normalize test y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_test), y_time_test.shape, np.unique(y_time_nor_test)))

    # ------------------------------------------- Set up VAE model and Start train process -------------------------------------------------
    _logger.info("Start training with epoch: {}. ".format(args.train_epoch_num))

    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    ## 打印模型的权重和偏置
    # for name, param in MyVAEModel.named_parameters():
    #     print(name, param)
    train_data = [[x_sc_train[:, i], y_time_nor_train[i], donor_index_train[i]] for i in range(x_sc_train.shape[1])]
    test_data = [[x_sc_test[:, i], y_time_nor_test[i], donor_index_test[i]] for i in range(x_sc_test.shape[1])]
    if batch_size is None:
        _logger.info("batch size is none, so don't set batch")
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=len(train_data), val_batch_size=len(test_data),
                                    test_batch_size=len(test_data), predict_batch_size=len(test_data),
                                    label_dic=label_dic)
    else:
        _logger.info("batch size is {}".format(batch_size))
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=batch_size, val_batch_size=batch_size,
                                    test_batch_size=batch_size, predict_batch_size=batch_size,
                                    label_dic=label_dic)
    # data.setup("train")
    experiment = VAEXperiment(MyVAEModel, config['exp_params'])

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # 2023-09-07 20:34:25 add check memory
    check_memory(max_attempts=max_attempts)
    device = auto_select_gpu_and_cpu(max_attempts=max_attempts)
    _logger.info("Auto select run on {}".format(device))

    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=args.train_epoch_num
                     )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, data)

    # test the model
    _logger.info("this epoch final, on test data:{}".format(runner.test(experiment, data)))
    # predict on train data
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    train_result = runner.predict(experiment, data_predict)
    train_clf_result, train_latent_mu_result, train_latent_log_var_result = train_result[0][0], train_result[0][1], \
        train_result[0][2]
    # predict on test data
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if (np.isnan(np.squeeze(test_clf_result)).any()):
        _logger.info("The Array contain NaN values")
    else:
        _logger.info("The Array does not contain NaN values")
    if test_clf_result.shape[1] == 1:
        # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
        _logger.info("predicted time of test donor is continuous.")
        try:
            predict_donors_dic[donor_list[fold]] = pd.DataFrame(data=np.squeeze(test_clf_result, axis=1),
                                                                index=sc_expression_test.index, columns=["pseudotime"])
        except:
            print("error here")
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donors_dic[donor_list[fold]] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                            index=sc_expression_test.index, columns=["pseudotime"])
    # acc = torch.tensor(torch.sum(labels_pred == y_time_nor_test).item() / (len(y_time_nor_test) * 1.0))
    # # ---------------------------------------------- plot sub result of training process for check  --------------------------------------------------
    if plot_trainingLossLine:
        _logger.info("Plot training loss line for check.")

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tags = EventAccumulator(tb_logger.log_dir).Reload().Tags()['scalars']
        _logger.info("All tags in logger: {}".format(tags))
        # Retrieve and print the metric results
        plot_tag_list = ["train_loss_epoch", "val_loss", "test_loss_epoch"]
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        plot_tag_list = ["train_clf_loss_epoch", "val_clf_loss", "test_clf_loss_epoch"]
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        plot_tag_list = ["train_Reconstruction_loss_epoch", "val_Reconstruction_loss", "test_Reconstruction_loss_epoch"]
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        try:
            plot_tag_list = ["train_batchEffect_loss_epoch", "val_batchEffect_loss", "test_batchEffect_loss_epoch"]
            plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        except:
            _logger.info("No batchEffect decoder.")
    else:
        _logger.info("Don't plot training loss line for check.")

    # # # ---------------------------------------------- plot sub latent space of sub model for check  --------------------------------------------------
    if plot_latentSpaceUmap:
        _logger.info("Plot each fold's UMAP of latent space for check.")
        umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_mu_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_mu_donor", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_logVar_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_logVar_donor", drop_batch_dim=0)
        try:
            umap_vae_latent_space(test_latent_mu_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_mu_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_mu_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_mu_donor", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_logVar_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_logVar_donor", drop_batch_dim=0)
        except:
            _logger.info("Too few test cells, can't plot umap of the latent space: {}.".format(len(y_time_nor_test)))
    else:
        _logger.info("Don't plot each fold's UMAP of latent space for check.")

    # ---------------------------------------------- save sub model parameters for check  --------------------------------------------------
    _logger.info(
        "encoder and decoder structure: {}".format({"encoder": MyVAEModel.encoder, "decoder": MyVAEModel.decoder}))
    _logger.info("clf-decoder: {}".format({"clf-decoder": MyVAEModel.clf_decoder}))
    torch.save(MyVAEModel, tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth')
    # _logger.info("detail information about structure save at： {}".format(tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth'))

    del MyVAEModel
    del runner
    del experiment
    # 清除CUDA缓存
    torch.cuda.empty_cache()
    return predict_donors_dic, test_clf_result, label_dic


def onlyTrain_model(sc_expression_df, donor_dic,
                    special_path_str,
                    cell_time,
                    time_standard_type, config, args, device=None, batch_dim=0, plot_latentSpaceUmap=True,
                    time_saved_asFloat=False,
                    batch_size=None, max_attempts=10000000, adversarial_bool=False, batch_dic=None,
                    donor_str="donor", time_str="time"):
    """
    use all donors as train data,
    then train a vae model with {latent dim} in latent space
    :param sc_expression_df:
    :param donor_dic:
    :param golbal_path:
    :param file_path:
    :param latent_dim:
    :param special_path_str:
    :param cell_time:
    :param time_standard_type:
    :param config:
    :param args:
    :param device:
    :param batch_dim:
    :param plot_latentSpaceUmap:
    :return:
    """
    from stemVAE.experiment import VAEXperiment
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict, SupervisedVAEDataset_onlyTrain
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pathlib import Path
    # _logger.info("calculate time cor gene. Use whole data retrain a model, predict average and medium change of each gene between normal-express and non-express")
    subFold_save_file_path = "{}{}/wholeData/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""),
                                                      special_path_str)
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    sc_expression_train = sc_expression_df.copy(deep=True)
    # we need to transpose data to correct its shape
    x_sc_train = torch.tensor(sc_expression_train.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))

    if time_saved_asFloat:  # 2023-08-06 13:25:48 trans 8.50000 to 850; 9.2500000 to 925; easy to  manipulate.
        cell_time_dic = dict(zip(cell_time.index, cell_time[time_str]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic) * 100).astype(int))
    elif time_standard_type == "organdic":
        cell_time_dic = dict(zip(cell_time.index, cell_time[time_str]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic)).astype(int))
    else:
        y_time_train = x_sc_train.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_train.index.values])

    # get each cell batch info
    if batch_dic is None:  # if no batch dic, just use donor id as batch index
        _logger.info("No batch dic, just use donor id as batch index")
        donor_index_train = x_sc_train.new_tensor(
            [int(donor_dic[cell_time.loc[_cell_name][donor_str]]) for _cell_name in sc_expression_train.index.values])
        batch_dic = donor_dic.copy()
    else:
        cell_donor_dic = dict(zip(cell_time.index, cell_time[donor_str]))
        donor_index_train = sc_expression_train.index.map(cell_donor_dic)
        donor_index_train = x_sc_train.new_tensor([int(batch_dic[_key]) for _key in donor_index_train])

    # for classification model with discrete time cannot use sigmoid and logit time type
    y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type)
    _logger.info("label dictionary: {}".format(label_dic))
    _logger.info(
        "Normalize train y_time_train type: {}, with y_time_train lable: {}, shape: {}, \nAfter trans y_time_nor_train detail: {}"
        .format(time_standard_type, np.unique(y_time_train), y_time_train.shape, np.unique(y_time_nor_train)))

    # ------------------------------------------- Set up VAE model and Start train process -------------------------------------------------
    _logger.info("Start training with epoch: {}. ".format(args.train_epoch_num))

    # if (int(config['model_params']['in_channels']) == 0) :
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    _logger.info("batch effect dic: {}".format(batch_dic))
    config['model_params']['batch_num'] = len(set(batch_dic.values()))

    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    train_data = [[x_sc_train[:, i], y_time_nor_train[i], donor_index_train[i]] for i in range(x_sc_train.shape[1])]
    if batch_size is None:
        _logger.info("batch size is none, so don't set batch")
        data = SupervisedVAEDataset_onlyTrain(train_data=train_data, train_batch_size=len(train_data), label_dic=label_dic)

    else:
        _logger.info("batch size is {}".format(batch_size))
        data = SupervisedVAEDataset_onlyTrain(train_data=train_data, train_batch_size=batch_size, label_dic=label_dic)

    experiment = VAEXperiment(MyVAEModel, config['exp_params'])
    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # add 2023-09-07 20:34:57 add memory check
    check_memory(max_attempts=max_attempts)
    device = auto_select_gpu_and_cpu(max_attempts=max_attempts)  # device: e.g. "cuda:0"
    _logger.info("Auto select run on {}".format(device))
    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="train_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=args.train_epoch_num
                     )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, data)

    # train data forward the model
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    train_result = runner.predict(experiment, data_predict)
    train_clf_result, train_latent_mu_result, train_latent_log_var_result = train_result[0][0], train_result[0][1], \
        train_result[0][2]

    """when we want to get an embedding for specific inputs: 
    We either
    1 Feed a hand-written character "9" to VAE, receive a 20 dimensional "mean" vector, then embed it into 2D dimension using t-SNE, 
    and finally plot it with label "9" or the actual image next to the point, or
    2 We use 2D mean vectors and plot directly without using t-SNE.
    Note that 'variance' vector is not used for embedding. 
    However, its size can be used to show the degree of uncertainty. 
    For example a clear '9' would have less variance than a hastily written '9' which is close to '0'."""
    if plot_latentSpaceUmap:
        if time_standard_type == "organdic":  # 2023-11-07 17:10:53 add for Joy project

            cell_type_number_dic = pd.Series(cell_time.broad_celltype_index.values, index=cell_time.broad_celltype).to_dict()
            umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, cell_type_number_dic, special_path_str, config,
                                  special_str="trainData_mu_time_broad_celltype", drop_batch_dim=0)
            umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                                  special_str="trainData_mu_donor_Dataset", drop_batch_dim=0)
            mesen_dic = {"False": 0, "True": 1}
            umap_vae_latent_space(train_latent_mu_result, y_time_nor_train.new_tensor(cell_time["mesen"]), mesen_dic, special_path_str, config,
                                  special_str="trainData_mu_donor_mesen", drop_batch_dim=0)
        else:
            umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, label_dic, special_path_str, config,
                                  special_str="trainData_mu_time", drop_batch_dim=0)
            umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                                  special_str="trainData_mu_donor", drop_batch_dim=0)
        # umap_vae_latent_space(train_latent_log_var_result, y_time_nor_train, label_dic, special_path_str, config,
        #                       special_str="trainData_logVar_time", drop_batch_dim=0)
        # umap_vae_latent_space(train_latent_log_var_result, donor_index_train, donor_dic, special_path_str, config,
        #                       special_str="trainData_logVar_donor", drop_batch_dim=0)
    train_latent_info_dic = {"mu": train_latent_mu_result,
                             "log_var": train_latent_log_var_result,
                             "label_true": y_time_nor_train,
                             "label_dic": label_dic,
                             "donor_index": donor_index_train,
                             "donor_dic": donor_dic, "total": train_result}
    if train_clf_result.shape[1] == 1:  # time is continues
        train_latent_info_dic["time_continues_bool"] = True
    else:
        train_latent_info_dic["time_continues_bool"] = False
    return sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, MyVAEModel, train_clf_result, label_dic, train_latent_info_dic


def read_model_parameters_fromCkpt(sc_expression_df, config_file, checkpoint_file, adversarial_bool=False):
    """
    read parameters or weights of model from checkpoint and predict on sc expression dataframe
    return the predict result
    :param sc_expression_df:
    :param config_file:
    :param checkpoint_file:
    :return:
    """
    import yaml

    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from stemVAE.experiment import VAEXperiment
    from stemVAE.dataset import SupervisedVAEDataset_onlyPredict
    # make data with gene express min
    x_sc = torch.tensor(sc_expression_df.values, dtype=torch.get_default_dtype()).t()
    data_x = [[x_sc[:, i], 0, 0] for i in range(x_sc.shape[1])]
    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['model_params']['in_channels'] = x_sc.shape[0]
    config['model_params']['batch_num'] = 2

    checkpoint = torch.load(checkpoint_file)
    # 去掉每层名字前面的 "model."
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        # 去掉前缀 "model."
        if key.startswith('model.'):
            key = key[6:]
        new_state_dict[key] = value
    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])
    MyVAEModel.load_state_dict(new_state_dict)
    MyVAEModel.eval()
    check_memory()
    device = auto_select_gpu_and_cpu()
    runner = Trainer(devices=[int(device.split(":")[-1])])
    seed_everything(config['exp_params']['manual_seed'], True)

    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=len(data_x))

    experiment = VAEXperiment(MyVAEModel, config['exp_params'])
    # z=experiment.predict_step(data_predict,1)
    train_result = runner.predict(experiment, data_predict)
    return train_result


def get_parameters_df(local_variables):
    """
    get local variables and return as dataframe
    to save as parameters file
    :param local_variables:
    :return:
    """
    variable_list = []

    # 遍历全局变量，过滤出用户自定义变量
    for var_name, var_value in local_variables.items():
        # 过滤掉特殊变量和模块
        if not var_name.startswith("_") and not callable(var_value):
            variable_list.append({'Variable Name': var_name, 'Value': var_value})

    # 创建 DataFrame，将存储的变量和值添加到 DataFrame 中
    df = pd.DataFrame(variable_list, columns=['Variable Name', 'Value'])
    return df


def predict_newData_preprocess_df(gene_dic, adata_new, min_gene_num, mouse_atlas_file):
    """
    2023-10-24 10:58:39
    as new data, predict the preprocess of new data, by concat with atlas data, and only return the preprocessed new data
    :param gene_dic:
    :param adata_new:
    :param min_gene_num:
    :param mouse_atlas_file:
    :return:
    """
    print("the original sc expression anndata should be gene as row, cell as column")

    try:
        mouse_atlas = anndata.read_csv(mouse_atlas_file, delimiter='\t')
    except:
        mouse_atlas = anndata.read_csv(mouse_atlas_file, delimiter=',')
    print("read the mouse atlas anndata with shape (gene, cell): {}".format(mouse_atlas.shape))
    mouse_atlas.obs_names = [gene_dic.get(name, name) for name in mouse_atlas.obs_names]
    print("change gene name of mouse atlas data to short gene name")
    # 查找adata1和adata2中duplicate columns, that is the duplicate cell name
    if len(set(adata_new.obs_names) & set(mouse_atlas.var_names)):
        print(
            f"Error: check the test data and mouse atlas data have cells with same cell name: {set(adata_new.obs_names) & set(mouse_atlas.var_names)}")
        return

    adata_concated = anndata.concat([mouse_atlas.copy(), adata_new.T.copy()], axis=1)
    print("merged sc data and external test dataset with shape (gene, cell): {}".format(adata_concated.shape))

    adata_concated = adata_concated.T  # 基因和cell转置矩阵
    print("Import data, cell number: {}, gene number: {}".format(adata_concated.n_obs, adata_concated.n_vars))

    # 数据数目统计
    sc.pp.filter_cells(adata_concated, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
    print("After cell threshold: {}, remain adata shape (cell, gene): {}".format(min_gene_num, adata_concated.shape))
    remain_test_cell_name = list(set(adata_concated.obs_names) & set(adata_new.obs_names))
    print(f"remain test adata cell num {len(remain_test_cell_name)}")
    sc.pp.normalize_total(adata_concated, target_sum=1e6)
    sc.pp.log1p(adata_concated)
    print("Finish normalize per cell, so that every cell has the same total count after normalization.")

    sc_expression_df = pd.DataFrame(data=adata_concated.X, columns=adata_concated.var_names, index=list(adata_concated.obs_names))

    denseM = sc_expression_df.values
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    print("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=sc_expression_df.columns, index=sc_expression_df.index)
    sc_expression_test_df = sc_expression_df.loc[remain_test_cell_name]
    loss_gene_shortName_list = list(set(mouse_atlas.obs_names) - set(adata_new.var_names))
    sc_expression_test_df[loss_gene_shortName_list] = 0
    sc_expression_test_df = sc_expression_test_df[mouse_atlas.obs_names]
    return sc_expression_test_df, loss_gene_shortName_list
