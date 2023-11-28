# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction 
@File    ：start_exps_trainOn18Donor_testOnRIF_downSample.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/18 20:55

nohup python -u start_exps_trainOn18Donor_testOnRIF_downSample.py >> logs/start_exps_trainOn18Donor_testOnRIF_downSample.log 2>&1 &
"""

import os, sys
import time
import argparse
import subprocess


def main():
    dataset_list = [
        # "preprocess_02_major_Anno0717_Gene0720",
        #             "preprocess_03_epi_Anno0717_Gene0720",
        #             "preprocess_05_NKT_Anno0717_Gene0720",
        #             "preprocess_06_myeloid_Anno0717_Gene0720",
                    "preprocess_07_fibro_Anno0717_Gene0720"]
    result_save_path = "230925_newAnno0717_Gene0720_trainOn18Donor_testOnRIF_downSample_dataAugmentation"

    epoch_num = 100
    python_file = "../VAE_fromDandan_testRifDonor_dataAugmentation.py"
    # python_file = "VAE_fromDandan_testOnRifDonor.py"
    time_type_list_re = ["neg1to1"]
    downsample_bool = "true"
    downsample_locationtype_list = ["row", "line", "col"]
    cmd_list = []

    print("start regression model")
    vae_param_file_list_re = ["supervise_vae_regressionclfdecoder"]
    for time_type in time_type_list_re:
        for downsample_locationtype in downsample_locationtype_list:
            for vae_param_file in vae_param_file_list_re:
                for dataset in dataset_list:
                    args = dict()
                    args["downsample_bool"] = downsample_bool
                    args["downsample_locationtype"] = downsample_locationtype
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
        print(f"--- Start: {i + 1}/{len(cmd_list)}")
        process = subprocess.Popen(cmd_list[i], shell=True)
        # os.system(cmd_list[i])
        pid = process.pid
        print(f"PID:{pid}")
        print(f"{cmd_list[i]}")
        time.sleep(60)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
