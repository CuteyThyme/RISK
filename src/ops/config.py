import argparse
from os.path import join
from collections import namedtuple

SOURCE_DIR = "dataset"

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])
FeverPairExample = namedtuple("FeverPairExample", ["id", "claim", "evidence", "label"])
PairExample = namedtuple("PairExample", ["id", "s1", "s2", "label"])
HardExample = namedtuple("HardExample", ["input_id", "attention_mask", "segment_id", "uncertainty", "label"])


FEVER_MAPS = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}
FEVER_LABELS = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]

NLI_LABELS = ["contradiction", "entailment", "neutral"]
QQP_LABELS = ['not duplicated', 'duplicated']   ### ['not_match', 'match']


HANS_SOURCE = join(SOURCE_DIR, "hans")
MULTINLI_SOURCE = join(SOURCE_DIR, "multinli")
FEVER_SOURCE = join(SOURCE_DIR, "fever")
QQP_PAWS_SOURCE = join(SOURCE_DIR, "qqp_paws")


LEX_BIAS_SOURCE = join("../biased_preds", "lex_overlap_preds.json")
HYPO_BIAS_SOURCE = join("../biased_preds", "hyp_only.json")

PRETRAINED_PATH = "/root/bert-base-uncased"

parser = argparse.ArgumentParser(description="Debiasing")


parser.add_argument('--reloss', action='store_true',help='reconstruct loss for autoencoder')
parser.add_argument('--reloss_type', default="mse", type=str, help='mse for reconstruct loss')
parser.add_argument('--loss_type', default="all", type=str, choices=["all", "rec", "rsr"])
parser.add_argument("--data_dir", default="dataset/", type=str, help="dir of dataset")
parser.add_argument("--dataset", default="mnli", type=str, choices=["mnli", "fever", "qqp"], help="debiasing task")
parser.add_argument("--seed", default=777, type=int, help="seed")
parser.add_argument("--log_dir", default="logs/", type=str, help="dir of dataset")
parser.add_argument('--p', default=1.0, type=float, help="p norm for reconstruction loss")
parser.add_argument('--lamda1', default=0.1, type=float, help="lamda1 for pca error")
parser.add_argument('--lamda2', default=0.1, type=float, help="lamda2 for projection error")
parser.add_argument('--intrinsic_dim', default=10, type=int, help="intrinsic dim of robust recovery layer")
parser.add_argument('--encoder_dim', default=128, type=int, help="output dim D for the encoder of autoencoder ")
parser.add_argument('--local_rank', type=int, default=-1, help="local gpu id")
parser.add_argument('--training_step', type=int, default=100, help="100 training iterations equal to one step")

parser.add_argument('--gpus', type=list, default=[0,1,2,3], help="local gpu id")
parser.add_argument('--gpu', type=int, default=1, help="local gpu id")
parser.add_argument('--nodes', type=int, default=1,  help='number of data loading workers (default: 4)')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

parser.add_argument("--max_seq_len", default=128, type=int, help="max sequence length")
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--optimizer', default="AdamW", type=str, help="optimizer for training ")
\
parser.add_argument('--world_size', default=4, type=int, help='number of nodes for distributed training')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--arch', metavar='ARCH', default='resnet18_with_table')
parser.add_argument('--best_model_name', type=str, default="model_best.pth.tar", help = '')