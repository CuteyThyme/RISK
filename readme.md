# RISK
This repo implements the main experiments of COLING 2022 paper:   
Less is Better: Recovering Intended-Feature Subspace to Robustify NLU Models

## Getting Started 

### Prerequisites:
- python 3.8.3
- pytorch 1.7.1
- transformers 4.23.1
- pretrained bert model [download_link](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz)

Official Links for data used in this paper:   
MNLI:  [https://cims.nyu.edu/~sbowman/multinli/](https://cims.nyu.edu/~sbowman/multinli/)     
HANS:  [https://github.com/tommccoy1/hans](https://github.com/tommccoy1/hans)    
ANLI:  [https://github.com/facebookresearch/anli](https://github.com/facebookresearch/anli)  

FEVER: [https://fever.ai/](https://fever.ai/)     
FEVER-Symmetric: [https://github.com/TalSchuster/FeverSymmetric](https://github.com/TalSchuster/FeverSymmetric)     

QQP:   [https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)   
PAWS:  [https://github.com/google-research-datasets/paws](https://github.com/google-research-datasets/paws)

For the convenience, you can download all the training data [here](https://drive.google.com/drive/folders/1aleJytl3SAKdGBsxZbxznwusINOnTAzh?usp=share_link).

We use NVIDIA GeForce RTX 2080 Ti for the experiments. Run the following to get the results in the paper.     
To be noted, considering the training convergence of the autoencoder, the training epochs should be set to 30.


```
# MNLI 
python src/main.py --gpu 0 --dataset mnli --batch_size 16 --epochs 30 --intrinsic_dim 25 --lamda1 0.002 --lamda2 0.002 --reloss --best_model_name mnli_model.pth.tar

# FEVER
python src/main.py --gpu 0 --dataset fever --batch_size 16 --epochs 30 --intrinsic_dim 30 --lamda1 0.009 --lamda2 0.009 --reloss --best_model_name fever_model.pth.tar

# QQP
python src/main.py --gpu 0 --dataset qqp --batch_size 16 --epochs 30 --intrinsic_dim 16 --lamda1 0.025 --lamda2 0.025 --reloss --best_model_name qqp_model.pth.tar
```

## Citation
The following is the bibtex for citation.
```
@inproceedings{wu-gui-2022-less,
    title = "Less Is Better: Recovering Intended-Feature Subspace to Robustify {NLU} Models",
    author = "Wu, Ting  and
      Gui, Tao",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.143",
    pages = "1666--1676"
}

```

