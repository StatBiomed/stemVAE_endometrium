
[//]: # (<div align="center">)

[//]: # (    <img src="images/stemVAE_logo.png" width = "350" alt="stemVAE">)

[//]: # (</div>)

# StemVAE: identify temporal information from endometrium cells via Deep generative model

[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit/) 
[![DOI](https://zenodo.org/badge/724489497.svg)](https://doi.org/10.5281/zenodo.13950075)
Contact: Yuanhua Huang, Dandan Cao, Yijun Liu

Email:  yuanhua@hku.hk.

## Introduction ()
StemVAE use the probabilistic latent space model to infer the pseudo-time of cells. StemVAE input consists of an mRNA expression matrix and real-time labels of cells, and output is the reconstruction of the expression matrix and predicted time. StemVAE, based on canonical variation atuo-encoder (VAE), includes an encoder, a cell-decoder, and a time-decoder. 

[//]: # (A preprint describing StemVAE's algorithms and results is at [bioRxiv]&#40;https://;.)



![](./stemVAE/231019model_structure.png)

---


## Contents

- [Latest Updates](#latest-updates)
- [Installations](#installation)
- [Usage](#usage)
    - [Model training](#model-training)
    - [Performance evaluation](#performance-evaluation)
    - [Spatial inference](#spatial-inference)
   

## Latest Updates
* v0.1 (Sep, 2023): Initial release.
---
## Installation
To install stemVAE, python 3.9 is required and follow the instruction
1. Install <a href="https://docs.conda.io/projects/miniconda/en/latest/" target="_blank">Miniconda3</a> if not already available.
2. Clone this repository:
```bash
  git clone https://github.com/awa121/stemVAE_endometrium
```
3. Navigate to `stemVAE_endometrium` directory:
```bash
  cd stemVAE_endometrium
```
4. (5-10 minutes) Create a conda environment with the required dependencies:
```bash
  conda env create -f environment.yml
```
5. Activate the `stemVAE` environment you just created:
```bash
  conda activate stemVAE
```
6. Install **pytorch**: You may refer to [pytorch installtion](https://pytorch.org/get-started/locally/) as needed. For example, the command of installing a **cpu-only** pytorch is:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Usage


StemVAE contains 2 main function: k fold test on dataset; predict on a new donor. And we also provide code to reproduce the result in the paper. 

To check available modules, run:
### prepare the preprocess data with R code:
```r  

# pre-define the location of preprocessed data 
# input the location of orginal .rds data
# Give a HVG gene load from a csv file (if you have)
savedir_sc_gene_table=data.frame(save_dir = c("data/preprocess_02_major_Anno0717_GeneVP0721/",
                                              "data/preprocess_03_epi_Anno0717_GeneVP0721/",
                                              "data/preprocess_05_NKT_Anno0717_GeneVP0721/",
                                              "data/preprocess_06_myeloid_Anno0717_GeneVP0721/",
                                              "data/preprocess_07_fibro_Anno0717_GeneVP0721/"),
                                 file_name = c("data/harmony.final.rds",
                                               "data/epi.harmony.rds",
                                               "data/nkt.harmony.rds",
                                               "data/mye.harmony.rds",
                                               "data/str.harmony.rds"),
                                 gene_file_name = c("data/major_vp_neg1to1_allvar.csv",
                                                    data/epi_vp_neg1to1_allvar.csv",
                                                    data/nkt_vp_neg1to1_allvar.csv",
                                                    data/mye_vp_neg1to1_allvar.csv",
                                                    data/str_vp_neg1to1_allvar.csv"))

for(index in 1: nrow(savedir_sc_gene_table))
{ 
  print(savedir_sc_gene_table[index,])
  save_dir = savedir_sc_gene_table[index,]$save_dir
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  } else {
    print("Dir already exists!")
  }
  print(save_dir)
  
  file_name=savedir_sc_gene_table[index,]$file_name
  print(paste("preprocess for ", file_name, sep = " "))
  
  
  data_orginal = readRDS(file_name)
  table(data_orginal$group)
  table(data_orginal$major_cell_type)
  table(data_orginal$cell_subtype)
  
  data_count = data_orginal[['RNA']]@counts
  data_count <- as.data.frame(data_count)
  
  data_nor = data_orginal[['RNA']]@data
  data_nor = as.data.frame(data_nor)
  
  sc_data_gene_list=rownames(data_count)
  
  gene_file_name=savedir_sc_gene_table[index,]$gene_file_name
  high_var_genes <- read.csv(gene_file_name, encoding="UTF-8")
  high_var_genes=high_var_genes$X[1:500]
  print(paste("HVG genes is got from file: ", gene_file_name,sep = ""))
  
  high_var_genes=intersect(sc_data_gene_list,high_var_genes)
  
  data_count_hvg = data_count[high_var_genes,]
  data_nor_hvg = data_nor[high_var_genes,]
  
  
  cellname_vs_time = data.frame(
    "cell" = colnames(data_orginal),
    "time" = data_orginal@meta.data$group,
    "donor"=data_orginal@meta.data$donor,
    "major_cell_type" = data_orginal@meta.data["major_cell_type"],
    "cell_subtype" = data_orginal@meta.data["cell_subtype"]
  )
  
  
  
  write.table(
    data_count_hvg,
    paste(save_dir, "data_count_hvg.csv", sep = "/"),
    row.names = TRUE,
    col.names = TRUE,
    sep = "\t"
  )
  write.table(
    data_nor_hvg,
    paste(save_dir, "data_nor_hvg.csv", sep = "/"),
    row.names = TRUE,
    col.names = TRUE,
    sep = "\t"
  )
  
  
  write.table(
    cellname_vs_time,
    paste(save_dir, "cell_with_time.csv", sep = "/"),
    row.names = FALSE,
    col.names = TRUE,
    sep = "\t"
  )
  print(paste("Save at", save_dir, sep = " "))
  print(paste("Finish preprocess for ", file_name, sep = " "))
}
```

### k fold test
The result will save to folder _results_, log file wile save to folder _logs_
```bash
python -u VAE_fromDandan_testOnOneDonor.py 
--vae_param_file=supervise_vae_regressionclfdecoder 
--file_path=preprocess_02_major_Anno0717_Gene0720 --time_standard_type=neg1to1 
--train_epoch_num=100 
--result_save_path=230728_newAnno0717_Gene0720_18donor_2type_plot 
> logs/log.log


```








