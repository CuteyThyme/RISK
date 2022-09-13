import time

import torch
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.matrix import accuracy, f1score
from utilis.meters import AverageMeter, ProgressMeter


def rsr_validate(val_loader, model, criterion, epoch=0, test=True, args=None, datasetname=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        f1scores = AverageMeter('F1 score', ':6.4f')
        if args.dataset == "qqp":
            progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5, f1scores],
            prefix='Test: ')
        else:
            progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        top5 = AverageMeter('Val Acc@5', ':6.2f')
        f1scores = AverageMeter('F1 score', ':6.4f')
        if args.dataset == "qqp":
            progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5, f1scores],
            prefix='Val: ')
        else:
            progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')

    model.eval()
    print('******************datasetname is {}******************'.format(datasetname))
    
    with torch.no_grad():
        end = time.time()
        for i, (input_ids, attention_masks, segment_ids, target) in enumerate(val_loader):

            input_ids = input_ids.cuda(args.gpu, non_blocking=True)
            attention_masks = attention_masks.cuda(args.gpu, non_blocking=True)
            segment_ids = segment_ids.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, rsrloss = model(input_ids, attention_masks, segment_ids)
            
            loss = criterion(output, target) + rsrloss

            acc1, acc5 = accuracy(output, target, topk=(1, 1), args=args, datasetname=datasetname)
            losses.update(loss.item(), input_ids.size(0))
            top1.update(acc1[0], input_ids.size(0))
            top5.update(acc5[0], input_ids.size(0))
            if args.dataset == "qqp":
                f1_score = f1score(output, target)
                f1scores.update(f1_score)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)
        
        if args.dataset == "qqp":
            print(' * Acc@1 {top1.avg:.3f} Acc@1 {top5.avg:.3f} F1-scores {f1scores.avg:.3f}'
                .format(top1=top1, top5=top5, f1scores=f1scores))
        else:
            print(' * Acc@1 {top1.avg:.3f} Acc@1 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
       
    return top1.avg