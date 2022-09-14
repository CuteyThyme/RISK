# RISK
This repo implements the main experiments of COLING 2022 paper:   
Less is Better: Recovering Intended-Feature Subspace to Robustify NLU Models

## Getting Started 

### Prerequisites:
- python
- pytorch
- transformers
- pretrained bert model [download_link](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz)

Links for data used in this paper, download and put them into dataset folder:   
MNLI:  [https://cims.nyu.edu/~sbowman/multinli/](https://cims.nyu.edu/~sbowman/multinli/)     
HANS:  [https://github.com/tommccoy1/hans](https://github.com/tommccoy1/hans)    
ANLI:  [https://github.com/facebookresearch/anli](https://github.com/facebookresearch/anli)  

FEVER: [https://fever.ai/](https://fever.ai/)     
FEVER-Symmetric: [https://github.com/TalSchuster/FeverSymmetric](https://github.com/TalSchuster/FeverSymmetric)     

QQP:   [https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)   
PAWS:  [https://github.com/google-research-datasets/paws](https://github.com/google-research-datasets/paws)

Run the following to get the results in the paper.
```
# MNLI 
python src/main.py --gpu 0 --dataset mnli --batch_size 16 --epochs 10 --intrinsic_dim 25 --lamda1 0.002 --lamda2 0.002 --reloss --best_model_name mnli_model.pth.tar

# FEVER
python src/main.py --gpu 0 --dataset fever --batch_size 16 --epochs 10 --intrinsic_dim 30 --lamda1 0.009 --lamda2 0.009 --reloss --best_model_name fever_model.pth.tar

# QQP
python src/main.py --gpu 0 --dataset qqp --batch_size 16 --epochs 10 --intrinsic_dim 16 --lamda1 0.025 --lamda2 0.025 --reloss --best_model_name qqp_model.pth.tar
```

## Citation
The following is the bibtex for citation.

