import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel, RobertaModel

from utilis.dimension_reduction import PCA_svd

class BasicBertModel(nn.Module):
    def __init__(self, pretrained_path, bert_config, args, num_labels):
        super(BasicBertModel, self).__init__()
        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits
