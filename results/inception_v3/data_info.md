-----------------------
### Target:
Number of Images: 75392

**In webvision_a folder**

lr: 0.1
dec_freq=40
epochs=200
best prec: 0.8780
loss: 0.4771

*model_best_a.tar*

-----------------------

### Baseline:
Number of Images: 34596

**In webvision folder**

lr: 0.1
dec_freq=40
epochs=150
best prec: 0.8015
loss: 0.7965

*model_best_baseline.tar*

-----------------------
### q30 finetune on baseline (v1):
Number of Images: 55757

**In webvision folder**

lr_epoch_map = {0: 0.1, 20: 0.01, 60: 0.001}
epochs=90
best prec: 0.8645
loss: 0.5881

*model_best_q30.tar*

-----------------------
### q15 finetune on baseline (v1):
Number of Images: 64349

**In webvision folder**

lr_epoch_map = {0: 0.1, 20: 0.01, 60: 0.001}
epochs=90
best prec: 0.8835
loss: 0.5164

*model_best_q15.tar*

-----------------------
### q10 finetune on baseline (v1):
Number of Images: 67880

**In webvision folder**

lr_epoch_map = {0: 0.1, 20: 0.01, 60: 0.001}
epochs=90
best prec: 0.8900
loss: 0.4796

*model_best_q10.tar*

-----------------------
### q10 finetune on baseline (v2):
Number of Images: 67880

**In webvision folder**

lr_epoch_map = {0: 0.1, 30: 0.01, 80: 0.001, 110: 0.0001}
epochs=150
best prec: 0.8885
loss: 0.4607

*model_best_q10_v2.tar*


-----------------------
### q10 finetune on baseline (v3):
Number of Images: 67880

**In webvision folder**

lr_epoch_map = {0: 0.1, 30: 0.01, 60: 0.001, 120: 0.0001}
epochs=150
best prec: 0.8845
loss: 0.4857

*model_best_q10_v3.tar*

-----------------------
### q10_denos finetune on baseline:
Number of Images: 64364

**In webvision folder**

lr_epoch_map = {0: 0.1, 30: 0.01, 80: 0.001, 110: 0.0001}
epochs=
best prec: 
loss: 

**

-----------------------

### Q10 from scratch:
Number of Images: 67880

**In webvision_a folder**

lr_epoch_map = {0: 0.1, 40: 0.01, 90: 0.001, 150: 0.0001}
epochs=200
best prec: 0.8870
loss: 0.4580

*model_best_q10_scratch.tar*

-----------------------
### Target finetune on baseline (v1):
Number of Images: 34596

**In webvision_b folder**

lr_epoch_map = {0: 001, 50: 0.0001}
epochs=100
best prec: 0.8800
loss: 0.5214

*model_best_a_fine_baseline.tar*

-----------------------
### Target finetune on q30:
Number of Images: 34596

**In webvision_b folder**

lr_epoch_map = {0: 001, 50: 0.0001}
epochs=100
best prec:
loss:

*model_best_a_fine_q30.tar*

-----------------------



