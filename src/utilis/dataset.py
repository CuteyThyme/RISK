import torch
from torch.utils.data import Dataset

# MAX_SEQ_LENGTH = 128


class PairDatasets(Dataset):
    def __init__(self, examples, tokenizer, labels_type, args):
   
        self.label2id = {label: i for i, label in enumerate(labels_type)}
        self.len = len(examples)
        
        input_ids, attention_masks, segment_ids, label_ids = processSentences(tokenizer, examples, args)
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_masks[index], self.segment_ids[index], self.label_ids[index]

    def __len__(self):
        return self.len


class HardDatasets(Dataset):
    def __init__(self, examples):
   
        self.len = len(examples)
        self.input_ids, self.attention_masks, self.segment_ids, self.label_ids = hard_process(examples)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_masks[index], self.segment_ids[index], self.label_ids[index]

    def __len__(self):
        return self.len


def hard_process(examples):
    input_ids = []
    attention_masks = []
    segment_ids = []
    label_ids = []
    for example in examples:
        input_ids.append(example.input_id)
        attention_masks.append(example.attention_mask)
        segment_ids.append(example.segment_id) 
        label_ids.append(example.label)
    return input_ids, attention_masks, segment_ids, label_ids


def check_data(s1, s2, label, args, labels_type):
    if len(s1.split()) == 0 or len(s2.split()) == 0:
        return False
    elif label in labels_type:
        return True
    else:
        return False


def processSentences(tokenizer, examples, args):
    input_ids = []
    attention_masks = []
    segment_ids = []
    label_ids = []
    MAX_SEQ_LENGTH = args.max_seq_len

    for example in examples:
        tokens_1 = tokenizer.tokenize(example.s1)
        tokens_2 = None
        if example.s2:
            tokens_2 = tokenizer.tokenize(example.s2)
            _truncate_seq_pair(tokens_1, tokens_2, MAX_SEQ_LENGTH - 3)
        else:
            if len(tokens_1) > MAX_SEQ_LENGTH - 2:
                tokens_1 = tokens_1[:(MAX_SEQ_LENGTH - 2)]

        tokens = ["[CLS]"] + tokens_1 + ["[SEP]"]
        segment_id = [0] * len(tokens) 
        attention_mask = [1] * len(tokens)

        if tokens_2:
            tokens += tokens_2 + ["[SEP]"]
            segment_id += [1] * (len(tokens_2) + 1)
            attention_mask += [1] * (len(tokens_2) + 1)

        input_id = tokenizer.convert_tokens_to_ids(tokens)

        if len(input_id) < MAX_SEQ_LENGTH: 
            padding_length = MAX_SEQ_LENGTH - len(input_id)
            input_id = input_id + ([tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            segment_id = segment_id + ([0] * padding_length)
         
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        segment_ids.append(segment_id)
        label_ids.append(example.label)

    return input_ids, attention_masks, segment_ids, label_ids


def _truncate_seq_pair(tokens_1, tokens_2, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_1) + len(tokens_2)
        if total_length <= max_length:
            break
        if len(tokens_1) > len(tokens_2):
            tokens_1.pop()
        else:
            tokens_2.pop()


class Collate_function:
    def collate(self, batch):
        input_ids, attention_masks, segment_ids, targets = zip(*batch)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        return input_ids, attention_masks, segment_ids, targets

    def __call__(self, batch):
        return self.collate(batch)


