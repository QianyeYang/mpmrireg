# Overview
The source code for submitted paper "Cross-Modality Image Registration using a Training-Time Privileged Third Modality"


## Training
* Please check and set up the hyper-parameters in ``config.py``, and save your command line in a bash file, like those examples in ``scripts``.
* The source code can only use processed medical image data, the structure of the data files can be organized as follows:
```
|---data folder
        |---train
           |---patiend_1
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
cd ./scripts
sh [any of the bash file in it.]
```

## Testing
* While training, a experiment folder will generated in the ``./logs``, for example ``./logs/15.privileged``
* use following commandlines to do the test, the results will be printed and be saved as ``results.pkl`` in the corresponding experiment folders after the test.
```
python test.py ./logs/15.privileged/ [GPU-id]
```
