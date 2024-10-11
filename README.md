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

### Installing necessary libraries
'''
!pip3 install -q -U transformers==4.38.2
!pip3 install -q -U datasets==2.18.0
!pip3 install -q -U bitsandbytes==0.42.0
!pip3 install -q -U peft==0.9.0
!pip3 install -q -U trl==0.7.11
!pip3 install -q -U accelerate==0.27.2
'''
