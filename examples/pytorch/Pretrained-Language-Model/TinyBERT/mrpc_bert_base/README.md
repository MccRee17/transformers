---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
- f1
model-index:
- name: bert_base
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE MRPC
      type: glue
      args: mrpc
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.7794117647058824
    - name: F1
      type: f1
      value: 0.8519736842105263
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert_base

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the GLUE MRPC dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6443
- Accuracy: 0.7794
- F1: 0.8520
- Combined Score: 0.8157

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     | Combined Score |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|:--------------:|
| No log        | 0.43  | 50   | 0.5723          | 0.6838   | 0.8122 | 0.7480         |
| No log        | 0.87  | 100  | 0.5163          | 0.7574   | 0.8385 | 0.7979         |
| No log        | 1.3   | 150  | 0.5209          | 0.7647   | 0.8373 | 0.8010         |
| No log        | 1.74  | 200  | 0.5091          | 0.7721   | 0.8421 | 0.8071         |
| No log        | 2.17  | 250  | 0.5315          | 0.7549   | 0.8264 | 0.7906         |
| No log        | 2.61  | 300  | 0.6815          | 0.7770   | 0.8535 | 0.8152         |


### Framework versions

- Transformers 4.20.0.dev0
- Pytorch 1.11.0+cu102
- Datasets 2.3.2
- Tokenizers 0.12.1
