# Gemma_Sprint
<img src="https://img.shields.io/badge/Google-4285F4?style=for-the-badge&logo=Google&logoColor=white">
Google for Developers Machine Learning Bootcamp Korea 2024
I'll develop the LLM model to help coding by finetuning gemma2-2b-it

## 1. Setting the enviornment to develop
### 1-1. Log in Hugging Face
To use the model in hugging face, you should log in your hugging face with your tokken number
```
from huggingface_hub import notebook_login
notebook_login()
```

### 1-2. Installing necessary libraries
```
!pip3 install -q -U transformers==4.38.2
!pip3 install -q -U datasets==2.18.0
!pip3 install -q -U bitsandbytes==0.42.0
!pip3 install -q -U peft==0.9.0
!pip3 install -q -U trl==0.7.11
!pip3 install -q -U accelerate==0.27.2
```

### 1-3. Import modules
```
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
```

## 2. Preparing the dataset
### 2-1. Dataset Load
```
from datasets import load_dataset
dataset = load_dataset("llama-duo/gemma2b-coding-eval-by-claude3sonnet")
```
You can take the dataset from this link --> (https://huggingface.co/datasets/llama-duo/gemma2b-coding-eval-by-claude3sonnet)
