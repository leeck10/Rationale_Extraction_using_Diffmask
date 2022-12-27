# Rationale_Extraction_using_DIFFMASK

## Introduction

For the NLI and sentiment analysis tasks, the rationale for the prediction result can be extracted using the post-analysis model.
To utilize the extracted rationale, we propose a method for integrating rationale information using rationale embedding.
Rationale embedding is effective in improving the model's performance by helping the model's prediction process.
It was shown that the rationale extracted using the post-hoc analysis model has significant information.

### References

* 정영준, 이창기. 근거를 이용한 한국어 감성 분석, 제34회 한글 및 한국어 정보처리 학술대회, 2022

![image](https://user-images.githubusercontent.com/41266083/209643638-cdbd111c-8d9c-4f05-80c9-d376ca6e5e55.jpeg)

## Dependencies

  python>=3.6
  
  torch>=1.5.0
  
  pytorch-lightning==0.7.3
  
  transformers==2.9.0
  
  spacy==2.2.4
  
  torch-optimizer==0.0.1a9
  
  matplotlib>=3.1.1

## train classification

$ run_sst.sh

## train diffmask

$ run_sst_diffmask.sh

## rationale extraction

$ SentimentClassificationNSMCDiffMaskAnalysis.ipynb
