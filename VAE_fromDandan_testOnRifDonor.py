# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：vanillaVAE_fromDandan_trainOnAllData.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023-07-14 15:36:22

dandan re-annotation 02-major dataset, add RIF (Re-Implantation Fail) donor
02-major include: 18 donors + 10 RIF donors
use 18 donors train a VAE model, test on RIF donors to get pseudotime

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
                        default="testOnRifDonor",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="preprocess_07_fibro_Anno0717_Gene0720",
                        help="Dandan file path.")

    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25

    parser.add_argument('--train_epoch_num', type=int,
                        default="5",
                        help="Train epoch num")
    parser.add_argument('--time_standard_type', type=str,
                        default="neg1to1",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")

    # supervise_vae   supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder",
                        help="vae model parameters file.")
    # parser.add_argument('--dropout_donor', type=str,
    #                     default="no",  # LH7_1  or no
    #                     help="dropout a donor.")

    # 2023-08-18 18:54:12 add preprocess sc data parameters for downsample the total expression count in the test RIF donor
    parser.add_argument('--downsample_bool', type=str2bool,
                        default="False",  # LH7_1  or no
                        help="whether downsample test data.")
    parser.add_argument('--downsample_locationtype', type=str,
                        default="line",  # LH7_1  or no
                        help="downsample location type: line, row, or col.")

    # parser.add_argument('--dropout_batch_effect_dim', type=int, default=5,  # LH7_1  or no
    #                     help="Consider donor id as batch effect, set dropout batch effect dim.")

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
    if args.downsample_bool:
        logger_file = '{}/{}_dim{}_time{}_epoch{}_downSampleLocation{}.log'.format(_path, args.vae_param_file,
                                                                                   latent_dim, time_standard_type,
                                                                                   args.train_epoch_num,
                                                                                   args.downsample_locationtype)
    else:
        logger_file = '{}/{}_dim{}_time{}_epoch{}.log'.format(_path, args.vae_param_file, latent_dim,
                                                              time_standard_type,
                                                              args.train_epoch_num,
                                                              )

    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))
    device = auto_select_gpu_and_cpu()
    _logger.info("Auto select run on {}".format(device))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # --------ff-------------------------------- 10-fold test RIF, each time train on 18 donor and test on 1 RIF donor-----------------------------------------------------
    cell_time = pd.read_csv(data_golbal_path + "/" + cell_info_file_csv, sep="\t", index_col=0)
    donor28 = list(set(cell_time["donor"]))
    train_donor_list = [_ for _ in donor28 if "LH" in _]
    train_donor_list = sorted(train_donor_list, key=LHdonor_resort_key)
    train_donor_dic = dict()
    for i in range(len(train_donor_list)):
        train_donor_dic[train_donor_list[i]] = i
    _logger.info("Consider donor as batch effect, donor use label: {}".format(train_donor_dic))

    test_donor_list = [_ for _ in donor28 if "RIF" in _]
    test_donor_list = sorted(test_donor_list, key=RIFdonor_resort_key)
    predict_donors_dic = dict()
    # for fold in range(4):
    for fold in range(len(test_donor_list)):
        _logger.info("test donor is {}".format(test_donor_list[fold]))
        # -----------------------------------------Preprocess data------------------------------------------------------
        special_path_str = "_" + args.time_standard_type
        drop_out_donor = test_donor_list[:fold] + test_donor_list[fold + 1:]
        sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv,
                                                                                    KNN_smooth_type, cell_info_file_csv,
                                                                                    drop_out_donor=drop_out_donor,
                                                                                    donor_attr="donor",
                                                                                    downSample_on_testData_bool=args.downsample_bool,
                                                                                    test_donor=test_donor_list[fold],
                                                                                    downSample_location_type=args.downsample_locationtype,
                                                                                    plot_boxPlot_bool=True,
                                                                                    special_path_str=special_path_str)
        _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))

        # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
        sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time["donor"] != test_donor_list[fold]]]
        sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time["donor"] == test_donor_list[fold]]]

        # # ---------------------------------------------- use all data to train a model and identify time-cor gene --------------------------------------------------
        sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, _ = onlyTrain_model(
            sc_expression_train,
            train_donor_dic,
            special_path_str,
            cell_time,
            time_standard_type, config, args,
            device=device,
            plot_latentSpaceUmap=False)
        # -------------------------------------------------- test on new dataset --------------------------------------
        predict_donors_dic, test_clf_result, _ = test_on_newDonor(test_donor_list[fold], sc_expression_test, runner, experiment,
                                                                  predict_donors_dic)
        del runner
        del _m
        del experiment
        # 清除CUDA缓存
        torch.cuda.empty_cache()
    # # ---------------------------------------------- plot total result  --------------------------------------------------
    cell_time = pd.read_csv(data_golbal_path + "/" + cell_info_file_csv, sep="\t", index_col=0)
    if test_clf_result.shape[1] == 1:
        _logger.info("plot predicted time of each test donor is continuous.")
        plot_on_each_test_donor_continueTime_windowTime(result_save_path, data_path, test_donor_list, predict_donors_dic,
                                                        latent_dim,
                                                        time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                                        model_name=config['model_params']['name'], cell_time=cell_time,
                                                        special_path_str=special_path_str,
                                                        plot_subtype_str="major_cell_type", special_str="_major_cell_type")
        plot_on_each_test_donor_continueTime_windowTime(result_save_path, data_path, test_donor_list, predict_donors_dic,
                                                        latent_dim,
                                                        time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                                        model_name=config['model_params']['name'], cell_time=cell_time,
                                                        special_path_str=special_path_str,
                                                        plot_subtype_str="cell_subtype", special_str="_cell_subtype")

        plot_on_each_test_donor_continueTime(result_save_path, data_path, test_donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="major_cell_type", special_str="_major_cell_type")
        plot_on_each_test_donor_continueTime(result_save_path, data_path, test_donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="cell_subtype", special_str="_cell_subtype")
    elif time_standard_type == "labeldic":
        _logger.info("plot predicted time of each test donor is discrete: {}.".format(label_dic))
        plot_on_each_test_donor_discreteTime(result_save_path, data_path, test_donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="major_cell_type", special_str="_major_cell_type")
        plot_on_each_test_donor_discreteTime(result_save_path, data_path, test_donor_list, predict_donors_dic, latent_dim,
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
