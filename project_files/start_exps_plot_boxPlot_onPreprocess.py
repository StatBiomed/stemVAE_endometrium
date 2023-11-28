# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：start_exps_plot_boxPlot_onPreprocess.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/20 23:23
nohup python -u start_exps_plot_boxPlot_onPreprocess.py > logs/start_exps_plot_boxPlot_onPreprocess.log 2>&1 &

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
    result_save_path = "230820_newAnno0717_plot_boxPlot_onPreprocess"

    epoch_num = 100
    python_file = "plot_boxPlot_onPreprocess0819.py"
    time_type_list_re = ["neg1to1"]

    cmd_list = []
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
                cmd = cmd + " > logs/" + "_".join(args.values()).replace("/", "_").replace(" ", "_") + ".log" + " 2>&1 &"
                cmd_list.append(cmd)
    for i in range(len(cmd_list)):
        print("#--- Start: {}/{}".format(i, len(cmd_list)))
        print(cmd_list[i])
        os.system(cmd_list[i])
        time.sleep(60)
        if (i + 1) % 3 == 0:
            time.sleep(60 * 10)
        if (i + 1) % 6 == 0:
            time.sleep(60 * 10)
        if (i + 1) % 9 == 0:
            time.sleep(60 * 10)

        sys.stdout.flush()


if __name__ == "__main__":
    main()
