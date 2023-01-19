# Rationale_Extraction_using_DIFFMASK

## Introduction

For the NLI and sentiment analysis tasks, the rationale for the prediction result can be extracted using the post-analysis model.
To utilize the extracted rationale, we propose a method for integrating rationale information using rationale embedding.
Rationale embedding is effective in improving the model's performance by helping the model's prediction process.
It was shown that the rationale extracted using the post-hoc analysis model has significant information.

![image](https://user-images.githubusercontent.com/41266083/209643638-cdbd111c-8d9c-4f05-80c9-d376ca6e5e55.jpeg)

### References

* 정영준, 이창기. 근거를 이용한 한국어 감성 분석, 제34회 한글 및 한국어 정보처리 학술대회, 2022

## Dependencies

* **python>=3.6**
* **pytorch>=1.5**: https://pytorch.org
* **pytorch-lightning==0.7.3**: https://pytorch-lightning.readthedocs.io
* **transformers>=2.9.0**: https://github.com/huggingface/transformers
* **torch-optimizer>=0.0.1a9**: https://github.com/jettify/pytorch-optimizer
* **matplotlib>=3.1.1**: https://matplotlib.org

## Installation

```bash
$ cd transformers-2.9.0
$ pip install ./
```
