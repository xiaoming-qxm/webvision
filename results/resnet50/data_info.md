-----------------------
### Target:
Number of Images: 75392

**In webvision_a folder**

lr: 0.1
dec_freq=40
epochs=200
best prec: 0.8515
loss: 0.5612

*model_best_a.tar*

-----------------------

### Baseline:
Number of Images: 34596

**In webvision_a folder**

lr: 0.1
dec_freq=40
epochs=150
best prec: 0.7365
loss: 1.0158

*model_best_baseline.tar*

-----------------------
### Target finetune on baseline (v1):
Number of Images: 34596

**In webvision_b folder**

lr_epoch_map = {0: 005, 50: 0.0005}
epochs=100
best prec: 
loss:

*model_best_a_fine_baseline.tar*

-----------------------
### Target finetune on q30 (v1):
Number of Images: 55757

**In webvision_c folder**

lr_epoch_map = {0: 005, 50: 0.0005}
epochs=100
best prec: 
loss:

*model_best_a_fine_q30.tar*



### Q10 from scratch:
Number of Images: 67880

**In webvision_a folder**

lr_epoch_map = {0: 0.1, 40: 0.01, 90: 0.001, 150: 0.0001}
epochs=200
best prec: 0.8395
loss: 0.6209

*model_best_q10_scratch.tar*

-----------------------

-----------------------
### q30 finetune on baseline (v1):
Number of Images: 55757

**In webvision folder**

lr_epoch_map = {0: 0.1, 20: 0.01, 60: 0.001}
epochs=90
best prec: 0.8090
loss: 0.7384

*model_best_q30.tar*

-----------------------
### q15 finetune on baseline (v1):
Number of Images: 64349

**In webvision folder**

lr_epoch_map = {0: 0.1, 20: 0.01, 60: 0.001}
epochs=90
best prec: 
loss:

*model_best_q15.tar*

-----------------------



