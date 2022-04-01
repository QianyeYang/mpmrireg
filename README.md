# Overview
The source code for submitted paper "Cross-Modality Image Registration using a Training-Time Privileged Third Modality"


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
* While training, a experiment folder will generated in the ``./logs/mpmrireg/``, for example ``./logs/mpmrireg/05-6.pri_gmi0.15_l2n1000_sample5``
* use following commandlines to do the test, the results will be printed and be saved as ``results.pkl`` in the corresponding experiment folders after the test.
```
python test.py ./logs/mpmrireg/05-6.pri_gmi0.15_l2n1000_sample5 [GPU-id]
```
* An example is provided to demonstrate how to use our model to test on an public data set from [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/QIN-PROSTATE-Repeatability) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13cGaVu8i0LSP-OHVz_eTp-Cfg8LxUJcy?usp=sharing). However, we still recommend our users to train/test the real world clinical data via GPUs.(Under construction)