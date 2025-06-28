# [IJCAI 2025] Leveraging MLLM Embeddings and Attribute Smoothing for Compositional Zero-Shot Learning  (Trident)

<!--要改的地方：arxiv version两处的链接，IJCAI version的链接-->

* **Authors**: [Xudong Yan](https://github.com/xud-yan), [Songhe Feng](https://faculty.bjtu.edu.cn/8407/), Yang Zhang, Jian Yang, Yueguan Lin, Haojun Fei
* **Affiliations**: Beijing Jiaotong University, Qifu Technology
* **More details**: [[arXiv version]](https://arxiv.org/abs/2411.12584) | [[code]](https://github.com/xud-yan/Trident) | [[IJCAI version]]()



Note: The supplementary material is provided in the paper's [arXiv version](https://arxiv.org/abs/2411.12584).



## 📝 Overview

**TL;DR**: We employ both LLM and MLLM to guide attribute-object disentanglement by generating auxiliary attributes and representing primitive words for CZSL, respectively.

<img src=".\images\overview.png" style="zoom: 50%;" />



## ⚙️ Setup

Our work is implemented in PyTorch framework. Create a conda environment `trident` using:

```
conda create --name trident python=3.8.0
conda activate trident
pip install -r requirements.txt
```



## ⬇️ Download

**Datasets**: In our work, we conduct experiments on three datasets: MIT-States, C-GQA, and VAW-CZSL. For VAW-CZSL, you can download this dataset from [this website](https://drive.google.com/drive/folders/1CalwDXkkGALxz0e-aCFg9xBmf7Pu4eXL?usp=sharing). For MIT-States and C-GQA, please using:

```
bash utils/download_data.sh
```



**Pre-trained models**: ViT-Large-Patch14-336px (the backbone) can be downloaded [here](https://huggingface.co/openai/clip-vit-large-patch14-336). LLaVA-v1.5-7b can be found [here](https://huggingface.co/liuhaotian/llava-v1.5-7b).

## 🏋️ Training

1. Before training Trident, please obtain the auxiliary attributes by GPT-3.5 through OpenAI official API, and get the last hidden states of LLaVA v1.5 **offline**， which can be found in utils folder.

2. Train Trident model with a specified configure file using:

   ```
   python train.py --cfg config/{DATASET_NAME}.yml
   ```

   

## 📊 Evaluation

Evaluate Trident model using:

```
python test.py --cfg config/{DATASET_NAME}.yml --load TRIDENT_MODEL.pth
```



## 📚 Citation

If you find our work helpful, please cite our paper:

```
@inproceedings{Yan_2025_IJCAI,
   title={Leveraging MLLM Embeddings and Attribute Smoothing for Compositional Zero-Shot Learning},
   author={Yan, Xudong and Feng, Songhe and Zhang, Yang and Yang, Jian and Lin, Yueguan and Fei, Haojun},
   journal={arXiv preprint arXiv:2411.12584},
   year={2024}
}
```
The citation of IJCAI version will be released soon!


## 🙏 Acknowledgement

Thanks for the publicly available code of [OADis](https://github.com/nirat1606/OADis) and [LLaVA](https://github.com/haotian-liu/LLaVA).

## 📬 Contact

If you have any questions or are interested in collaboration, please feel free to contact me at xud_yan@163.com/xud_yan@bjtu.edu.cn .
