# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：VAE_fromDandan_testOnExternal_eachTestOnOneSample_newAnnor0717.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/3 17:43

2023-08-03 17:44:00
preprocess on 18donor+1testDonor(from a external dataset)
train on 18donor, test on 1 donor from one external dataset
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
from stemVAE import *
from utils.utils_plot import *


def main():
    parser = argparse.ArgumentParser(description="stemVAE train model on 18 donors and test on a external donor/dataset.")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="test/",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="preprocess_02_major_Anno0717_Gene0720/",
                        help="Dandan file path.")
    parser.add_argument('--external_test_path', type=str,
                        default="test_external0823/preprocess_A13_A30_visium/",
                        help="Dandan file path.")

    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25

    parser.add_argument('--train_epoch_num', type=int,
                        default="3",
                        help="Train epoch num")
    parser.add_argument('--time_standard_type', type=str,
                        default="labeldic",  # labeldic
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")

    # supervise_vae_discreteclfdecoder  /  supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_discreteclfdecoder",
                        help="vae model parameters file.")

    args = parser.parse_args()

    data_golbal_path = "data/"
    result_save_path = "results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    yaml_path = "model_configs/"
    # --------------------------------------- import vae model parameters from yaml file----------------------------------------------
    print(os.getcwd())
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
                                                          time_standard_type, args.train_epoch_num)
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))
    device = auto_select_gpu_and_cpu()
    _logger.info("Auto select run on {}".format(device))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # --------------------- donors number of external dataset -fold test RIF,
    # --------------------- each time train on 18 donor and test on 1 donor which from external dataset -------------------
    cell_time = pd.read_csv(data_golbal_path + "/" + cell_info_file_csv, sep="\t", index_col=0)
    donor28 = list(set(cell_time["donor"]))
    # get 18 donors for train
    train_donor_list = [_ for _ in donor28 if "LH" in _]
    train_donor_list = sorted(train_donor_list, key=LHdonor_resort_key)
    train_donor_dic = dict()
    for i in range(len(train_donor_list)):
        train_donor_dic[train_donor_list[i]] = i
    _logger.info("Consider donor as batch effect, donor use label: {}".format(train_donor_dic))

    # get 10 RIF donor need be droped
    drop_donorRIF_list = [_ for _ in donor28 if "RIF" in _]
    _logger.info("Preprocess dropout donor RIF: {}".format(drop_donorRIF_list))

    # get the donor info of external dataset
    external_sc_data_file_csv = args.external_test_path + "/data_count_hvg.csv"
    external_cell_info_file_csv = args.external_test_path + "/cell_with_time.csv"
    cell_time_external = pd.read_csv("{}/{}".format(data_golbal_path, external_cell_info_file_csv), sep="\t", index_col=0)

    test_donor_list = sorted(list(set(cell_time_external["donor"])))
    _logger.info(f"test donor id is {test_donor_list}.")
    predict_donors_dic = dict()
    # each time train a model by 18 donor, test on one donor from external dataset
    for fold in range(len(test_donor_list)):
        _logger.info("test donor is {}/{}: {}".format(fold + 1, len(test_donor_list), test_donor_list[fold]))
        # ----------------------------Preprocess data, drop RIF and others external donor.----------------------------------------
        drop_out_donor = drop_donorRIF_list + test_donor_list[:fold] + test_donor_list[fold + 1:]
        sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv,
                                                                                    KNN_smooth_type, cell_info_file_csv,
                                                                                    drop_out_donor=drop_out_donor,
                                                                                    donor_attr="donor",
                                                                                    external_file_name=external_sc_data_file_csv,
                                                                                    external_cell_info_file=external_cell_info_file_csv)
        _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))
        special_path_str = "_" + args.time_standard_type
        # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
        sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time["donor"] != test_donor_list[fold]]]
        sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time["donor"] == test_donor_list[fold]]]

        # ---------------------------------------------- use all data to train a model --------------------------------------------------
        sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, train_latent_info_dic = onlyTrain_model(
            sc_expression_train, train_donor_dic,
            special_path_str,
            cell_time,
            time_standard_type, config, args,
            device=device,
            plot_latentSpaceUmap=False, batch_size=50000)
        # -------------------------------------------------- test on new dataset --------------------------------------
        predict_donors_dic, test_clf_result, test_latent_info_dic = test_on_newDonor(test_donor_list[fold], sc_expression_test,
                                                                                     runner, experiment,
                                                                                     predict_donors_dic)
        del runner
        del _m
        del experiment
        # 清除CUDA缓存
        torch.cuda.empty_cache()

    # --------------------------------------- save all results ----------------------------------------------------------
    save_result_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    result_dic_json_file_name = "{}/{}_testOnExternal_{}.json".format(save_result_path, config['model_params']['name'],
                                                                      args.external_test_path.replace("/", "_").replace(".rds", "").replace(
                                                                          " ", "_"))
    import json
    save_dic = dict()
    for _donor, val in predict_donors_dic.items():
        save_dic[_donor + "_pseudotime"] = list(val["pseudotime"].astype(float))
        save_dic[_donor + "_cellid"] = list(val.index)

    with open(result_dic_json_file_name, 'w') as f:
        json.dump(save_dic, f)  # 2023-07-03 22:31:50
    _logger.info("Finish save clf result at: {}".format(result_dic_json_file_name, ))

    #  ---------------- plot total result directly from just saved json file --------------------------------------------------

    plot_time_window_fromJSON(json_file=result_dic_json_file_name)
    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
