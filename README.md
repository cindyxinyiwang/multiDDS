# Balancing Training for Multilingual Neural Machine Translation

Implementation of the paper
>[Balancing Training for Multilingual Neural Machine Translation](https://arxiv.org/pdf/2004.06748.pdf)

>Xinyi Wang, Yulia Tsvetkov, Graham Neubig

### Data:
The preprocessed and binarized data for fairseq can be downloaded >[here](https://drive.google.com/file/d/1xNlfgLK55SbNocQh7YpDcFUYymfVNEii/view?usp=sharing)
To process data from scrach, see the script
```
util_scripts/prepare_multilingual_data.sh
```

### Training Scripts:
The training scripts for many-to-one translation of the related language group (Related M2O) is under the directory ```job_scripts/related_ted8_m2o/```.

Our methods:
MultiDDS-S: 
```bash
job_scripts/related_ted8_m2o/multidds_s.sh 
```
MultiDDS: 
```bash 
job_scripts/related_ted8_m2o/multidds.sh 
``` 

Baselines:
Proportional: 
```bash 
job_scripts/related_ted8_m2o/proportional.sh 
``` 
Temperature: 
```bash 
job_scripts/related_ted8_m2o/temperature.sh 
```

The scripts for Related O2M is under the directory ```job_scripts/related_ted8_o2m/``` 

The scripts for Diverse M2O is under the directory ```job_scripts/diverse_ted8_m2o/``` 

The scripts for Diverse O2M is under the directory ```job_scripts/diverse_ted8_o2m/``` 

### Inference Scripts:
Each of the experiment script directory contains a trans.sh file to translate the test set. To translate the test set for the Related M2O MultiDDS-S 
```bash
job_scripts/related_ted8_m2o/trans.sh checkpoints/related_ted8_m2o/multidds_s/ 
``` 

To translate other experiment, simply replace the argument with the experiment checkpoint directory.

# Citation

Please cite as:

```bibtex
@inproceedings{wang2020multiDDS,
  title = {Balancing Training for Multilingual Neural Machine Translation},
  author = {Xinyi Wang, Yulia Tsvetkov, Graham Neubig},
  booktitle = {ACL},
  year = {2020},
}
```
