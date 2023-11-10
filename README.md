# Overview
The source code for the submitted paper "Cross-Modality Image Registration using a Training-Time Privileged Third Modality".


## Setup

* Better using virtual environment to avoid conflicts. For example:
  ```
  conda create -n mpmrireg python=3.7
  # ... after installation

  conda activate mpmrireg
  git clone https://github.com/QianyeYang/mpmrireg.git
  cd mpmrireg
  pip install -r requirments.txt
  pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  ```
  This demo is tested on Ubuntu 18.04 (Nvidia GPU required), but the training/testing code should be compatible with Windows as well.

## Training
* Please check and set up the hyper-parameters in ``./config/global_train_config.py`` and ``./config/mpmrireg_train_config.py``. Save your command line in a bash file, like those examples in ``./scripts/mpmrireg``.
* The source code can only use processed medical image data, the structure of the data files can be organized as follows:
```
|---./data/mpmrireg/
        |---train
           |---patient_1
              |---t2.npy
              |---dwi.npy
              |---dwi_b0.npy
              |---dwi_ldmk_1.npy ... dwi_ldmk_n.npy
              |---t2_ldmk_1.npy ... t2_ldmk_n.npy
           |---patient_2
           .
           .
           |---patient_n
        |---test
           |---...(structure same as above)
        |---val
           |---...(structure same as above)
```
* use following commandlines to repeat the experients in the paper
```
sh ./scripts/mpmrireg/[any of the bash file in it]
```

## Testing

### Data cleaning and preprocessing
* The link to the public data used in this paper is [here](https://wiki.cancerimagingarchive.net/display/Public/QIN-PROSTATE-Repeatability). A specific tool ([NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)) is required to download the data. We also provided the corresponding .tcia file ``./data/QIN-PROSTATE-Repeatability_v2_20180510.tcia``, which can be directly used to download the data via this tool. Please manually move the downloaded folder into the ``./data`` folder. 
* Use the following command to do the data cleaning. The purpose of this step is to find the proper data for doing inference, e.g., to find the patients who have both T2w and DWIs. The path ``./data/Cancer_Image_Archive_Selected_Data`` will be generated to perserve the cleaned samples.
  ```
  cd ./data
  python CIA_cleaning.py
  ``` 
* [ITK_SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) was used to manually convert the dicom format in to nifty format. The cleaned and format-converted data can be directly accessed [here](https://drive.google.com/file/d/1CXFLT2Bdvnjd5Zb9MmpFIknoRX1JHQPx/view?usp=sharing) to save some time. We recommend to directly use the converted data to avoid unexpectable failures, such as the conflict cause by the data update and structure adjustment on the server of the Cancer Imaging Archive. Please manually unzip the .zip file in the ``./data`` folder.
* Then use the following command line to do the preprocessing. Then the processed data will be generated to ``./data/CIA-external-npy``.
  ```
  python CIA_preprocessing.py 
  ```
* The fully-preprocessed data are also provided [here](https://drive.google.com/file/d/15l4IBfNUTdOwQL6rY2H6ekpwfaeNIfPj/view?usp=sharing), if the original links are not available. For those users who want to skip the data preprocessing can also download via this link. Please manually download and unzip it under the ``./data`` folder.

### Inference
* While training, a experiment folder will generated in the ``./logs/mpmrireg/``, for example ``./logs/mpmrireg/05-6.pri_gmi0.15_l2n1000_sample5``
* use following commandlines to do the inference, the results will be printed and be saved as ``results.pkl`` in the corresponding experiment folders after the test.
```
python test.py ./logs/mpmrireg/05-6.pri_gmi0.15_l2n1000_sample5 [GPU-id]
```
* An example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13cGaVu8i0LSP-OHVz_eTp-Cfg8LxUJcy?usp=sharing) is provided to demonstrate how to use our model to test on an public data set from the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/QIN-PROSTATE-Repeatability). To keep the simplicity of this demo and because of the data limitation, we only used a few samples from this public data set. However, the users could manually upload the rest of the data, or choose to mount their google drives to this Colab environment to access the data, if they are interested with more cases.

* Please note that this demo is only working on CPUs. We recommend our users to train/test the real world clinical data via GPUs, in order to get faster training/testing speed.


## Feedbacks
* Please be free to create issues in this repo and to let me know if there're any problems. Thanks!