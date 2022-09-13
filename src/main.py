import os
import random
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.optim import AdamW
import torch.utils.data
import torch.distributed as dist

from utilis.load_data import load_mnli, load_hans, load_fever, load_qqp_paws
from utilis.dataset import PairDatasets
from utilis.dataset import Collate_function
from ops import config
# import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from training.schedule import lr_setter
from training.train import rsr_train
from training.validate import rsr_validate
from utilis.meters import AverageMeter
from utilis.saving import save_checkpoint
from ops.config import parser
from models.rsr_bert import RSRBertModel

from transformers import AutoConfig, AutoTokenizer

best_acc = 0

## huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        

def main():
    args = parser.parse_args()
    
    if args.dataset == "qqp":
        args.classes_num = 2
    else:
        args.classes_num = 3
    
    set_random_seed(args.seed)
    args.log_path = os.path.join(args.log_dir, args.dataset, "logs.txt")
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
  
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    

    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_acc

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    bert_config = AutoConfig.from_pretrained(config.PRETRAINED_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_PATH)
    model = RSRBertModel(config.PRETRAINED_PATH, bert_config, args, args.classes_num)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.classes_num)
    nn.init.xavier_uniform_(model.classifier.weight, .1)
    nn.init.constant_(model.classifier.bias, 0.)

    if args.gpu is not None: 
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if args.gpu is not None:
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
     
    if args.dataset == "mnli":
        train_examples = load_mnli(mode="train")
        val_examples = load_mnli(mode="match_dev")
        test_examples = load_hans()

    elif args.dataset == "fever":
        train_examples = load_fever(mode="train")
        val_examples = load_fever(mode="dev")
        test_examples = load_fever(mode="symmv1_generated")
    
    elif args.dataset == "qqp":
        train_examples = load_qqp_paws(mode="qqp_train")
        val_examples = load_qqp_paws(mode="qqp_dev")
        test_examples = load_qqp_paws(mode="paws_devtest")

    train_dataset = PairDatasets(train_examples, tokenizer, config.NLI_LABELS, args)
    val_dataset = PairDatasets(val_examples, tokenizer, config.NLI_LABELS, args)
    test_dataset = PairDatasets(test_examples, tokenizer, config.NLI_LABELS, args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

   
    # begin to train
    for epoch in range(args.start_epoch, args.epochs):

        rsr_train(train_loader, model, criterion, optimizer, epoch, args)
        val_acc = rsr_validate(val_loader, model, criterion, epoch, False, args, datasetname=args.dataset)
        test_acc = rsr_validate(test_loader, model, criterion, epoch, True, args, datasetname=args.dataset)

        is_best = val_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            # pass
            print('Saving...')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.best_model_name)

      

if __name__ == '__main__':
    main()
