# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：VAE_fromDandan_testOnOneDonor.py.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/7/16 17:52 

Test VAE model on new annotation 02-major data (/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/preprocess_02_major_Anno0731)
use "major_cell_type"	"cell_subtype" plot sub-image of test result

dataset with 18 donors
use 17 donors to train a VAE model, use 1 donor to test the performance.
"""

import os

import torch

torch.set_float32_matmul_precision('high')
import pyro

from utils.logging_system import LogHelper

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
from utils.utils_project import *
from collections import Counter
import os
import yaml
import argparse
from utils.utils_plot import *


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="test",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="preprocess_03_epi_Anno0717_GeneVP0721",
                        help="Dandan file path.")

    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25

    parser.add_argument('--train_epoch_num', type=int,
                        default="2",
                        help="Train epoch num")
    parser.add_argument('--time_standard_type', type=str,
                        default="neg1to1",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")

    # supervise_vae            supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder",
                        help="vae model parameters file.")

    args = parser.parse_args()

    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"
    result_save_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    # save_yaml_config(vars(args), path='{}/config.yaml'.format(data_golbal_path + data_path))
    yaml_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/"
    # --------------------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ---------------------------------------set logger and parameters, creat result save path and folder----------------------------------------------

    latent_dim = config['model_params']['latent_dim']
    KNN_smooth_type = args.KNN_smooth_type

    time_standard_type = args.time_standard_type
    sc_data_file_csv = data_path + "/data_count_hvg.csv"
    cell_info_file_csv = data_path + "/cell_with_time.csv"

    _path = '{}/{}/'.format(result_save_path, data_path)
    if not os.path.exists(_path):
        os.makedirs(_path)
    logger_file = '{}/{}_dim{}_time{}_epoch{}.log'.format(_path, args.vae_param_file, latent_dim,
                                                          time_standard_type, args.train_epoch_num,
                                                          )
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # -----------------------------------------Preprocess data, drop RIF donor------------------------------------------------------
    cell_time = pd.read_csv(data_golbal_path + "/" + cell_info_file_csv, sep="\t", index_col=0)
    donor28 = list(set(cell_time["donor"]))
    drop_donor_list = [_ for _ in donor28 if "RIF" in _]
    _logger.info("Preprocess dropout donor: {}".format(drop_donor_list))
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv,
                                                                                KNN_smooth_type, cell_info_file_csv,
                                                                                drop_out_donor=drop_donor_list,
                                                                                donor_attr="donor")

    special_path_str = "_dropOutRIF" + "_" + args.time_standard_type  # LH7_1
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["donor"])
    donor_list = sorted(donor_list, key=LHdonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    _logger.info("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))
    # result_test_time_pesudo = dict()
    # ---------------------------------------- 18-fold test -----------------------------------------------------
    predict_donors_dic = dict()
    for fold in range(len(donor_list)):
        predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, donor_list, sc_expression_df, donor_dic,
                                                                      batch_dic,
                                                                      special_path_str, cell_time, time_standard_type,
                                                                      config, args,
                                                                      plot_trainingLossLine=False,
                                                                      plot_latentSpaceUmap=False)
        predict_donors_dic.update(predict_donor_dic)
    # # ---------------------------------------------- plot total result  --------------------------------------------------
    if test_clf_result.shape[1] == 1:
        _logger.info(
            "plot predicted time of each test donor is continuous with shape: {}; and time style is {}.".format(
                test_clf_result.shape, time_standard_type))
        plot_on_each_test_donor_continueTime_windowTime(result_save_path, data_path, donor_list, predict_donors_dic, latent_dim,
                                                        time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                                        model_name=config['model_params']['name'], cell_time=cell_time,
                                                        special_path_str=special_path_str,
                                                        plot_subtype_str="major_cell_type", special_str="_major_cell_type")
        plot_on_each_test_donor_continueTime_windowTime(result_save_path, data_path, donor_list, predict_donors_dic, latent_dim,
                                                        time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                                        model_name=config['model_params']['name'], cell_time=cell_time,
                                                        special_path_str=special_path_str,
                                                        plot_subtype_str="cell_subtype", special_str="_cell_subtype")
        plot_on_each_test_donor_continueTime(result_save_path, data_path, donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="major_cell_type", special_str="_major_cell_type")
        plot_on_each_test_donor_continueTime(result_save_path, data_path, donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="cell_subtype", special_str="_cell_subtype")
    elif time_standard_type == "labeldic":
        _logger.info(
            "plot predicted time of each test donor is discrete labeldic with shape: {}; and label dict: {}.".format(
                test_clf_result.shape, label_dic))
        plot_on_each_test_donor_discreteTime(result_save_path, data_path, donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="major_cell_type", special_str="_major_cell_type")
        plot_on_each_test_donor_discreteTime(result_save_path, data_path, donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="cell_subtype", special_str="_cell_subtype")
    else:
        _logger.info("Error in label")
        exit(1)


    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
