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
model-index:
- name: bert_base
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE RTE
      type: glue
      args: rte
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.6101083032490975
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert_base

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the GLUE RTE dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8070
- Accuracy: 0.6101

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
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 0.64  | 50   | 0.6878          | 0.6209   |
| No log        | 1.28  | 100  | 0.6911          | 0.5307   |
| No log        | 1.92  | 150  | 0.6708          | 0.6065   |
| No log        | 2.56  | 200  | 0.8580          | 0.6029   |


### Framework versions

- Transformers 4.20.0.dev0
- Pytorch 1.11.0+cu102
- Datasets 2.3.2
- Tokenizers 0.12.1
