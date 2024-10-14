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

```
train_data = dataset['gemma2b_coding_gpt4o_100k_by_claude3sonnet']
```

## 3. Finetuning Gemma
### 3-1. Adjusting Prompt for learning
```
def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['instructions'])):
        prompt_list.append(r"""<bos><start_of_turn>user
다음 명령에 따라 코드를 작성해 주세요:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['instructions'][i], example['target_responses'][i]))
    return prompt_list
```

### 3.2 Setting QLoRA
    lora_config = LoraConfig(
    r=6,
    lora_alpha = 8,
    lora_dropout = 0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
