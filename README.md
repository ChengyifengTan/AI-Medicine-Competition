
# 数据构建代码说明

该代码用于构建适配 MedM-VL 多模态模型训练格式的数据集。

### 1. 推荐项目结构

```
.
├── script/                  
    ├── build_dataset.py     # 数据集构建代码
    ├── build.sh             # 数据集构建脚本
    ├── split.py             # 切分开发集的样例代码
    ├── templates.json       # 各任务的prompt模板
    ├── test_predict.json    # 样例的提交格式，同时作为测试集标签（标签都是0）
    └── evaluate.py          # 结果评估代码            
└── data/ 
    ├── bounding_box_train.json  # 全部数据的bounding box信息
    ├── train.json           # 全部数据的标签
    ├── train_label.json     # 训练集标签（非定位任务）
    ├── dev_label.json       # 开发集标签（非定位任务）
    ├── train_box.json       # 训练集定位框（定位任务）
    ├── dev_box.json         # 开发集定位框（定位任务）
    ├── dataset/             # 输出数据目录
    └── mri_images/          # 图片数据总目录
        ├── train/            
    	└── test/          
```

推荐的项目代码库结构如上，其中train_label.json、dev_label.json为切分train.json得到的训练集和开发集，train_box.json、dev_box.json为切分bounding_box_train.json的定位信息训练集和开发集。

### 2. build.sh参数信息

| 参数                  | 类型      | 默认值                                     | 描述                                                         |
| --------------------- | --------- | ------------------------------------------ | ------------------------------------------------------------ |
| `--task_type`         | list[str] | `["qd","sl","zjppt","zyzg","positioning"]` | 要构建的任务类型，包括：`qd`、`sl`、`zjppt`、`zyzg`、`positioning` |
| `--train_images_path` | str       | `/data/mri_images/train`                   | 训练图像的根目录                                             |
| `--test_images_path`  | str       | `/data/mri_images/train`                   | 测试图像的根目录                                             |
| `--train_label_path`  | str       | `/data/train_label.json`                   | 非定位任务的训练标签文件路径                                 |
| `--test_label_path`   | str       | `/data/dev_label.json`                     | 非定位任务的测试标签文件路径                                 |
| `--train_box`         | str       | `/data/train_box.json`                     | 定位任务的训练框标注文件路径                                 |
| `--test_box`          | str       | `/data/dev_box.json`                       | 定位任务的测试框标注文件路径                                 |
| `--output_folder`     | str       | `/data/dataset`                            | 输出生成的 JSON 数据集的目录                                 |
| `--sag_image`         | list[int] | `[6]`                                      | 所使用的矢状面图像编号，[5，6]会同时给予模型两张矢状位图像   |
| `--sag_type`          | str       | `""`                                       | 设置为 `seperate` 时，会为每个矢状切片单独构建数据，如sag_image为[5,6,7]时相当于单独执行[5]、[6]、[7]（不会影响定位任务） |
| `--suffix`            | str       | `""`                                       | 输出文件名的后缀                                             |
| `--type_dataset`      | str       | `"both"`                                   | 可选 `train`、`dev`、`test` 或 `both`，指定构建数据集的形式，both代表`train`和`dev` |

### 3. split.py切分开发集代码

该代码简单的划分了训练集和开发集，可以参考修改

### 4. 需要提交的内容

不要提交构建的开发集的答案！

实际的测试集数据是mri_images/test中的图像数据，标签信息在data/test_predict.json中，该文件同时为提交文件的样例，其中所有标签都被设置为了0。

### 5. evaluate.py评估脚本

调用示例如下，需要替换标签位置和预测生成的答案位置

文件格式应和data/test_predict.json一致

python evaluate.py --label_path "../data/test_label.json" --answer_path "./output/answer.json"

### 6. 参考训练数据构建脚本

（1）切分数据，得到训练集和开发集数据

运行python split.py

（2）构建训练集和开发集

曲度和顺列任务数据较少，推荐使用矢状位5、6、7三张图片分别构建训练数据，定位数据同样推荐使用更多数据

```
python build_dataset.py \
    --task_type '["qd","sl","positioning"]' \
    --train_images_path "../data/mri_images/train" \
    --test_images_path "../data/mri_images/train" \
    --train_label_path "../data/train_label.json" \
    --test_label_path "../data/dev_label.json" \
    --train_box "../data/train_box.json" \
    --test_box "../data/dev_box.json" \
    --output_folder "../data/dataset" \
    --sag_image "[5,6,7]" \
    --sag_type "seperate" \
    --suffix "" \
    --type_dataset "train"
```

椎间盘膨突和中央椎管任务数据较多，可以只用正中一张图片构建数据

```
python build_dataset.py \
    --task_type '["zjppt","zyzg"]' \
    --train_images_path "../data/mri_images/train" \
    --test_images_path "../data/mri_images/train" \
    --train_label_path "../data/train_label.json" \
    --test_label_path "../data/dev_label.json" \
    --train_box "../data/train_box.json" \
    --test_box "../data/dev_box.json" \
    --output_folder "../data/dataset" \
    --sag_image "[6]" \
    --sag_type "" \
    --suffix "" \
    --type_dataset "train"
```

（3）构建开发集

```
python build_dataset.py \
    --task_type '["qd","sl","zjppt","zyzg"]' \
    --train_images_path "../data/mri_images/train" \
    --test_images_path "../data/mri_images/train" \
    --train_label_path "../data/train_label.json" \
    --test_label_path "../data/dev_label.json" \
    --train_box "../data/train_box.json" \
    --test_box "../data/dev_box.json" \
    --output_folder "../data/dataset" \
    --sag_image "[6]" \
    --sag_type "" \
    --suffix "" \
    --type_dataset "dev"
```

（4）构建测试集

这里简单使用矢状位正中一张图片

```
python build_dataset.py \
    --task_type '["qd","sl","zjppt","zyzg"]' \
    --train_images_path "../data/mri_images/train" \
    --test_images_path "../data/mri_images/test" \
    --train_label_path "../data/train_label.json" \
    --test_label_path "../data/test_predict.json" \
    --train_box "../data/train_box.json" \
    --test_box "" \
    --output_folder "../data/dataset" \
    --sag_image "[6]" \
    --sag_type "" \
    --suffix "" \
    --type_dataset "test"
```


# MedM-VL: What Makes a Good Medical LVLM?

[![arXiv](https://img.shields.io/badge/Arxiv-2504.04323-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2504.04323) [![hf_space](https://img.shields.io/badge/🤗-%20Open%20In%20HF-blue.svg)](https://huggingface.co/collections/shiym2000/medm-vl-67f739e50d344d712eb7b010) [![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE)

![architecture](./assets/architecture.png)

MedM-VL is a **modular**, LLaVA-based codebase for medical LVLMs, supporting flexible customization of encoders, connectors, and LLMs.

MedM-VL focuses on **small-scale** medical LVLMs, designed for **direct deployment** in real-world medical scenarios or **efficient fine-tuning** on downstream tasks.


## :newspaper: News

+ **[2025.04.10]**: The model weights (v1.0) have been uploaded to Hugging Face.
  + [shiym2000/MedM-VL-2D-3B-en · Hugging Face](https://huggingface.co/shiym2000/MedM-VL-2D-3B-en)
  + [shiym2000/MedM-VL-CT-Chest-3B-en · Hugging Face](https://huggingface.co/shiym2000/MedM-VL-CT-Chest-3B-en)
  + [shiym2000/MedM-CLIP-CT · Hugging Face](https://huggingface.co/shiym2000/MedM-CLIP-CT)
+ **[2025.04.06]**: The technical report has been released on arXiv.
  + [\[2504.04323\] MedM-VL: What Makes a Good Medical LVLM?](https://arxiv.org/abs/2504.04323)
+ **[2024.12.19]**: The complete code has been released on GitHub.


## :sparkles: Features

MedM-VL (v1.0: single image input, more details on Hugging Face)
+ [shiym2000/MedM-VL-2D-3B-en · Hugging Face](https://huggingface.co/shiym2000/MedM-VL-2D-3B-en): Trained on **2D** medical images and **English** medical texts.
+ [shiym2000/MedM-VL-CT-Chest-3B-en · Hugging Face](https://huggingface.co/shiym2000/MedM-VL-CT-Chest-3B-en): Trained on **3D** chest CT volumes and **English** medical texts.


## :package: Installation

``` bash
# 1. clone and navigate
git clone https://github.com/MSIIP/MedM-VL.git
cd MedM-VL

# 2. create a conda environment, activate it and install packages
conda create -n medm python=3.10
conda activate medm
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


## :rocket: Getting Started

If you are confused about some parameters during usage, please refer to [Parameter Interpretation](docs/param_interpretation.md).

### 1. Train a general medical LVLM from scratch

``` bash
# For 2D medical LVLMs
# 1. pre-train (annotation format: docs/example_2d_pretrain.json)
bash scripts/train/MedM-VL-2D/pretrain_en.sh
# 2. fine-tune (annotation format: docs/example_2d_finetune.json)
bash scripts/train/MedM-VL-2D/finetune_en.sh

# For 3D medical LVLMs
# 1. pre-train (annotation format: docs/example_3d_pretrain.json)
bash scripts/train/MedM-VL-CT-Chest/pretrain_en.sh
# 2. fine-tune (annotation format: docs/example_3d_finetune.json)
bash scripts/train/MedM-VL-CT-Chest/finetune_en.sh

# In fact, there is no difference in the annotation file format between
# pre-training and fine-tuning. The former is from image-text pairs
# while the latter refers to instruction tuning data.
```

### 2. Fine-tune a specialized medical LVLM with pre-trained weights

``` bash
# For 2D medical LVLMs
# 1. download weights from Hugging Face
pip install -U huggingface_hub
huggingface-cli download --resume-download shiym2000/MedM-VL-2D-3B-en --local-dir work_dirs/MedM-VL-2D-3B-en
# 2. fine-tune using LoRA (annotation format: docs/example_2d_finetune.json)
bash scripts/train/finetune_2d.sh

# For 3D medical LVLMs
# 1. download weights from Hugging Face
pip install -U huggingface_hub
huggingface-cli download --resume-download shiym2000/MedM-VL-CT-Chest-3B-en --local-dir work_dirs/MedM-VL-CT-Chest-3B-en
# 2. fine-tune using LoRA (annotation format: docs/example_3d_finetune.json)
bash scripts/train/finetune_3d.sh

# You can choose full or LoRA fine-tuning based on available GPU memory.
```

### 3. Inference

``` bash
# For 2D medical LVLMs
# inference (annotation format: docs/example_2d_inference.json)
bash scripts/eval/inference_2d.sh

# For 3D medical LVLMs
# inference (annotation format: docs/example_3d_inference.json)
bash scripts/eval/inference_3d.sh

# Compared to `finetune.json``, `conversations` in `inference.json` lacks
# the final response, which will be generated by the model.
```

### 4. Demo

``` bash
# Launch a Gradio demo locally.
bash scripts/playground.sh
```


## :robot: Model Zoo

<table>
  <tr align="center">
    <td><b>Encoder</b></td>
    <td><b>Connector</b></td>
    <td><b>LLM</b></td>
  </tr>
  <tr valign="top">
    <td>
      <li><a href="https://arxiv.org/abs/2103.00020"> CLIP (2021) </a></li>
      <li><a href="https://arxiv.org/abs/2303.15343"> SigLIP (2023) </a></li>
      <li><a href="https://arxiv.org/abs/2404.00578"> M3D-CLIP (2023) </a></li>
      <li><a href="https://huggingface.co/collections/shiym2000/medm-clip-67f7afd8a3dbcff656466805"> MedM-CLIP <a></li>
    </td>
    <td>
      <li> MLP </li>
      <li> Spatial Pooling </li>
      <li> Attention Pooling </li>
    </td>
    <td>
      <li><a href="https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/"> Phi-2 (2023) </a></li>
      <li><a href="https://arxiv.org/abs/2404.14219"> Phi-3 (2024) </a></li>
      <li><a href="https://arxiv.org/abs/2412.15115"> Qwen2.5 (2024) </a></li>
      <li><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/"> Llama-3.2 (2024) </a></li>
    </td>
  </tr>
</table>


## :book: Citation

``` bibtex
@article{shi2025medm,
  title={MedM-VL: What Makes a Good Medical LVLM?},
  author={Shi, Yiming and Yang, Shaoshuai and Zhu, Xun and Wang, Haoyu and Li, Miao and Wu, Ji},
  journal={arXiv preprint arXiv:2504.04323},
  year={2025}
}
```


## :heart: Acknowledgements

We would like to express our gratitude to the following resources:
+ [**TinyLLaVA_Factory**](https://github.com/TinyLLaVA/TinyLLaVA_Factory) - An open-source modular codebase for small-scale large multimodal models (LMMs).
