## Introduction
This repository is about fine-tuning LLaMA 3.2-3B, 1B models in the SDE project(UNITN). The dataset is [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA), I use the pqa_unlabeled subset (61.2k rows).

## Live Demo
[Colab](https://colab.research.google.com/drive/1oQZRVqmxLZmd36pSVwJg_c2wVRchFZMo?usp=sharing) 

Use free T4 GPU.

Thanks to 4-bit quantization, running LLaMA-3B requires only about 2.5 GB of GPU memory, and running LLaMA-3B requires only about 1.3 GB of GPU memory. Therefore, if your GPU memory is at least 4 GB, these models can run on your computer.

## Fine-tuning models weights
[LLaMA 3.2-3B](https://huggingface.co/Avalon-S/llama3_3B_pubmedqa_sde)
[LLaMA 3.2-1B](https://huggingface.co/Avalon-S/llama3_1B_pubmedqa_sde)

## Installation
```
git clone --depth 1 https://github.com/Avalon-S/LLaMA-Factory-SDE.git
cd LLaMA-Factory-SDE
pip install -e ".[torch,metrics]"
```
If you encounter library dependency and version issues, please refer to the [installation tutorial](README_LF.md)|[中文安装教程](README_LF_zh.md) of LLaMA-Factory.

## Usage
Fine-tune first, then merge LoRA weights. 
The first code file will generate a folder containing Lora weights, which need to be merged into the original LLaMA model during inference.

The second code file will merge the two into a new model. At this time, you can directly load the new model to complete the inference.
### Fine-tune LLaMA 3.2-3B on PubMedQA dataset
```
python ft_llama_3B.py
python merge_3B.py
```
### Fine-tune LLaMA 3.2-3B on PubMedQA dataset
```
python ft_llama_1B.py
python merge_1B.py
```
### Chat with model
```
python test.py
python test_merged.py
```

## Acknowledgements
This repository is a fork of the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) project, licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0). 

We greatly appreciate the original authors' efforts and contributions, which laid the foundation for this work. 
