# MOSS-RLHF

**Congratulations**🎉🎉🎉 We received **the best paper award** at NIPS 2023 Workshop on Instruction Tuning and Instruction Following!

庆祝🎉🎉🎉, "Secrets of RLHF in Large Language Models Part I: PPO" 荣获 NIPS 2023 Workshop on Instruction Tuning and Instruction Following 最佳论文奖！！！

<p align="center" width="100%">
<a href="https://arxiv.org/abs/2307.04964" target="_blank"><img src="./assets/img/moss.png" alt="MOSS" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-brightgreen.svg)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-blue.svg)](./DATA_LICENSE)
[![Model License](https://img.shields.io/badge/Model%20License-GNU%20AGPL%203.0-red.svg)](./MODEL_LICENSE)

### *MOSS-RLHF `<br>`👉 `<a href="https://openlmlab.github.io/MOSS-RLHF/" target="_blank">`[Home page]*

### *"Secrets of RLHF in Large Language Models Part I: PPO" `<br>`👉 `<a href="https://arxiv.org/abs/2307.04964" target="_blank">`[Technical report I]`</a>`*

### *"Secrets of RLHF in Large Language Models Part II: Reward Modeling" `<br>`👉 `<a href="https://arxiv.org/abs/2401.06080" target="_blank">`[Technical report II]`</a>`*

## 🌟🌟🌟 Breaking News

👉 Mon, 15. January 2024. We have released the code for training the reward model and the annotated hh-rlhf dataset([hh-rlhf-strength-cleaned)](https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned "https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned")!

👉 Fri, 12. January 2024. We have released the second paper **"Secrets of RLHF in Large Language Models Part II: Reward Modeling"**!

## 🌟 News

👉 Wed, 12. July 2023. We have released Chinese reward model based OpenChineseLlama-7B!
[moss-rlhf-reward-model-7B-zh](https://huggingface.co/Ablustrund/moss-rlhf-reward-model-7B-zh/tree/main)
`<br>`

👉 Thu, 13. July 2023. We have released English reward model and SFT model based Llama-7B!
[moss-rlhf-reward-model-7B-en](https://huggingface.co/fnlp/moss-rlhf-reward-model-7B-en)

[moss-rlhf-sft-model-7B-en](https://huggingface.co/fnlp/moss-rlhf-sft-model-7B-en)
`<br>`

👉 Wait a minute ! Thu, 14. July 2023. We have released English policy model after aligning with RLHF!
[moss-rlhf-policy-model-7B-en](https://huggingface.co/fnlp/moss-rlhf-policy-model-7B-en)
`<br>`

## 🧾 Open-source List

### RL related

- [X] Open source code for RL training in large language models.
- [X] A 7B Chinese reward model based on openChineseLlama.
- [X] A 7B English reward model based on Llama-7B.
- [X] SFT model for English.
- [X] Policy model for English after RLHF.

### RM related

- [X] Open source code for reward model training in large language models.
- [X] HH-RLHF dataset with preference strength annotation.
- [X] HH-RLHF validation set cleaned by GPT-4.

- ...

## 🌠 Introduction

Due to the challenges of reward design, environment interaction, and agent training, coupled with huge trial and error cost of large language models, there is a significant barrier for AI researchers to motivate the development of technical alignment and safe landing of LLMs. The stable training of RLHF has still been a puzzle.
In this technical report, we intend to help researchers to train their models stably with human feedback.

Contributions are summarized as follows:

1) We release competitive Chinese and English reward models, respectively, which have good cross-model generalization ability, alleviating the cost of relabeling human preference data;
2) We conduct in-depth analysis on the inner workings of PPO algorithm and propose the PPO-max algorithm to ensure stable model training;
3) We release the complete PPO-max codes to ensure that the LLMs in the current SFT stage can be better aligned with humans.

<div align="center" width="100%">
<img style="width: 80%; min-width: 500px; display: block; margin: auto; margin-bottom: 20px" alt="MOSS-RLHF" src="./assets/img/img1.jpg">
</div>

<div align="center" width="100%">
<img style="width: 80%; min-width: 500px; display: block; margin: auto; margin-bottom: 20px" alt="MOSS-RLHF" src="./assets/img/img2.jpg">
</div>

## 🔩 Requirements & Setup

This repository works on Python 3.8 and PyTorch 1.13.1.

We recommend using the **conda** virtual environment to run the code.

#### Step 1: Create a new Python virtual environment

```bash
conda update conda -n base -c defaults
conda create -n rlhf python=3.8
conda activate rlhf
```

#### Step 2: Install PyTorch and TensorBoard

```bash
conda install pytorch==1.13.1 pytorch-cuda=11.7 tensorboard -c pytorch -c nvidia
```

#### Step 3: Install the remaining dependencies

```bash
conda install datasets accelerate safetensors chardet cchardet -c huggingface -c conda-forge
pip3 install transformers sentencepiece einops triton==1.0.0 rouge jionlp==1.4.14 nltk sacrebleu cpm_kernels

apt install libaio-dev
DS_BUILD_OPS=1 pip install deepspeed
```

## ✨ Start training your own model!

### Training PPO model

Run code in a few steps.

#### Step 1: Recover Reward model weights

We can not directly release the full weight of the reward model because of protocol restrictions.
You can merge the diff weight with original Llama-7B to recover the reward model we used.

We upload the diff models, thanks to tatsu-lab, you can recover the reward model follow these steps:

```bash
1) Download the weight diff into your local machine. The weight diff is located at:
# For English:
# SFT model
https://huggingface.co/fnlp/moss-rlhf-sft-model-7B-en
# Reward model
https://huggingface.co/fnlp/moss-rlhf-reward-model-7B-en
# Policy model
https://huggingface.co/fnlp/moss-rlhf-policy-model-7B-en

# For Chinese:
https://huggingface.co/Ablustrund/moss-rlhf-reward-model-7B-zh/tree/main

2) Merge the weight diff with the original Llama-7B:
# For English:
# Reward model
python merge_weight_en.py recover --path_raw decapoda-research/llama-7b-hf --path_diff ./models/moss-rlhf-reward-model-7B-en/diff --path_tuned ./models/moss-rlhf-reward-model-7B-en/recover --model_type reward
# SFT model
python merge_weight_en.py recover --path_raw decapoda-research/llama-7b-hf --path_diff ./models/moss-rlhf-sft-model-7B-en/diff --path_tuned ./models/moss-rlhf-sft-model-7B-en/recover --model_type sft
# Policy model
python merge_weight_en.py recover --path_raw decapoda-research/llama-7b-hf --path_diff ./models/moss-rlhf-policy-model-7B-en/diff --path_tuned ./models/moss-rlhf-policy-model-7B-en/recover --model_type policy
# For Chinese:
python merge_weight_zh.py recover --path_raw decapoda-research/llama-7b-hf --path_diff ./models/moss-rlhf-reward-model-7B-zh/diff --path_tuned ./models/moss-rlhf-reward-model-7B-zh/recover
```

#### Step 2: Select your own SFT model.

Because of some limitations, we can not release the **Chinese** SFT model (Currently).
You can use your own SFT model, or a strong base model instead of our SFT model.

#### Step 3: Start training

Run the command below.

```
# For Chinese:
# You need to use your own sft model currently.
bash train_ppo_zh.sh

# For English:
# We have loaded the sft model and reward model to huggingface.
bash train_ppo_en.sh

```

### Training reward model

To train the reward model, you need to specify the initial model (`--hf_model_name_or_path`) for the reward model (e.g., meta-llama/Llama-2-7b-hf) and preference dataset(`--data_path`) (such as hh-rlhf, or you can use our provided [annotated hh-rlhf](https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned "https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned") which has a format consistent with the training code), and run the command below.

```
# annotated dataset: https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned
# Assuming you have specified the --hf_model_name_or_path and --data_path parameters.
bash train_rm.sh
```

## Citation

```bibtex
@article{zheng2023secrets,
      title={Secrets of RLHF in Large Language Models Part I: PPO}, 
      author={Rui Zheng and Shihan Dou and Songyang Gao and Wei Shen and Binghai Wang and Yan Liu and Senjie Jin and Qin Liu and Limao Xiong and Lu Chen and Zhiheng Xi and Yuhao Zhou and Nuo Xu and Wenbin Lai and Minghao Zhu and Rongxiang Weng and Wensen Cheng and Cheng Chang and Zhangyue Yin and Yuan Hua and Haoran Huang and Tianxiang Sun and Hang Yan and Tao Gui and Qi Zhang and Xipeng Qiu and Xuanjing Huang},
      year={2023},
      eprint={2307.04964},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
