# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：start_exps_newAnno0717.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/7/16 19:03


nohup python -u start_exps_newAnno0717.py > logs/start_exps_newAnno0717_230728_newAnno0717_GeneImmunity0720_18donor_2type_plot.log 2>&1 &
nohup python -u start_exps_newAnno0717.py > logs/start_exps_newAnno0717_230728_newAnno0717_Gene0720_18donor_2type_plot.log 2>&1 &
nohup python -u start_exps_newAnno0717.py > logs/start_exps_newAnno0717_230728_newAnno0717_GeneVP0721_18donor_2type_plot.log 2>&1 &
"""
import os, sys
import time
import yaml
import argparse


def main():
    # time.sleep(60*60*2)

    dataset_list = ["preprocess_02_major_Anno0717_Gene0720",
                    "preprocess_03_epi_Anno0717_Gene0720",
                    "preprocess_05_NKT_Anno0717_Gene0720",
                    "preprocess_06_myeloid_Anno0717_Gene0720",
                    "preprocess_07_fibro_Anno0717_Gene0720"]
    result_save_path = "230728_newAnno0717_Gene0720_18donor_2type_plot"

    # dataset_list = ["preprocess_02_major_Anno0717_GeneVP0721",
    #                 "preprocess_03_epi_Anno0717_GeneVP0721",
    #                 "preprocess_05_NKT_Anno0717_GeneVP0721",
    #                 "preprocess_06_myeloid_Anno0717_GeneVP0721",
    #                 "preprocess_07_fibro_Anno0717_GeneVP0721"]
    # result_save_path = "230728_newAnno0717_GeneVP0721_18donor_2type_plot"

    epoch_num = 100
    python_file = "../VAE_fromDandan_testOnOneDonor.py"
    time_type_list_re = ["neg1to1"]
    time_type_list_cl = ["labeldic"]  # "supervise_vae_noclfdecoder", "supervise_vae", can only use this type

    cmd_list = []
    vae_param_file_list_cl = ["supervise_vae"]
    print("#--- start program_index")
    # for classification model with discrete time cannot use sigmoid and logit time type
    for time_type in time_type_list_cl:
        for vae_param_file in vae_param_file_list_cl:
            for dataset in dataset_list:
                args = dict()
                args["vae_param_file"] = vae_param_file
                args["file_path"] = dataset
                args["time_standard_type"] = time_type
                args["train_epoch_num"] = str(epoch_num)
                args["result_save_path"] = result_save_path
                cmd = "nohup python -u " + python_file
                for key, value in args.items():
                    cmd = cmd + " --" + str(key) + "=" + str(value)
                cmd = cmd + " > logs/" + "_".join(args.values()).replace("/","_").replace(" ","_") + ".log" + " 2>&1 &"
                cmd_list.append(cmd)

    print("start regression model")
    vae_param_file_list_re = ["supervise_vae_regressionclfdecoder"]
    # time_type_list = ["log2", "0to1", "neg1to1", "sigmoid", "logit"]
    for time_type in time_type_list_re:
        for vae_param_file in vae_param_file_list_re:
            for dataset in dataset_list:
                args = dict()
                args["vae_param_file"] = vae_param_file
                args["file_path"] = dataset
                args["time_standard_type"] = time_type
                args["train_epoch_num"] = str(epoch_num)
                args["result_save_path"] = result_save_path
                cmd = "nohup python -u " + python_file
                for key, value in args.items():
                    cmd = cmd + " --" + str(key) + "=" + str(value)
                cmd = cmd + " > logs/" + "_".join(args.values()).replace("/","_").replace(" ","_") + ".log" + " 2>&1 &"
                cmd_list.append(cmd)
    for i in range(len(cmd_list)):
        print("#--- Start: {}/{}".format(i, len(cmd_list)))
        print(cmd_list[i])
        # os.system(cmd_list[i])
        # time.sleep(60)
        # if (i + 1) % 3 == 0:
        #     time.sleep(60 * 30)
        # if (i + 1) % 6 == 0:
        #     time.sleep(60 * 30)
        # if (i + 1) % 9 == 0:
        #     time.sleep(60 * 30)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
