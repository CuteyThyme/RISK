import os
import random
import shutil
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.autograd import Variable
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter

from transformers import get_linear_schedule_with_warmup

from torch import distributed as dist


def rsr_train(train_loader, model, criterion, optimizer, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    training_steps = (len(train_loader) - 1 / args.epochs + 1) * args.epochs
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps = 0.1 * training_steps,
        num_training_steps=training_steps
        )


    model.train()

    end = time.time()
    for i, (input_ids, attention_masks, segment_ids, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_ids = input_ids.cuda(args.gpu)
        attention_masks = attention_masks.cuda(args.gpu)
        segment_ids = segment_ids.cuda(args.gpu)
        target = target.cuda(args.gpu)

        output, rsrloss = model(input_ids, attention_masks, segment_ids)

        loss = criterion(output, target) + rsrloss

        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), input_ids.size(0))
        top1.update(acc1[0], input_ids.size(0))
        top5.update(acc5[0], input_ids.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()
         
        if i % args.print_freq == 0:
            method_name = args.log_path.split('/')[-2]
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)