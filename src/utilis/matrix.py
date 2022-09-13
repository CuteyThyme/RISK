import torch
from torch import distributed as dist
from sklearn.metrics import f1_score

def accuracy(output, target, topk=(1,), args=None, datasetname=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        if datasetname != None:
            if datasetname == 'HANS':
                tmp_zero = torch.zeros_like(pred).cuda(args.gpu, non_blocking=True)
                pred = torch.where(pred == 2, tmp_zero, pred).cuda(args.gpu, non_blocking=True)
        # print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def f1score(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        probs = torch.nn.functional.softmax(output, dim=1)
        preds = torch.argmax(probs, dim=1)
        f1_scores = f1_score(target.cuda().data.cpu().numpy(), preds.cuda().data.cpu().numpy()) * 100.0
        return f1_scores