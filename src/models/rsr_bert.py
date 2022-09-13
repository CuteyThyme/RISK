import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F


class RSRLayer(nn.Module):
    def __init__(self, d, D):
        super().__init__()
        self.d = d # 16
        self.D = D # 128
        self.A = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d, D)))

    def forward(self, z):
        # z is the output from the encoder
        z_hat = self.A @ z.view(z.size(0), self.D, 1)
        return z_hat.squeeze(2)


class RSRLoss(nn.Module):
    def __init__(self, lambda1, lambda2, d, D):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.d = d
        self.D = D
        self.register_buffer(
            "Id", torch.eye(d)
        )

    def forward(self, z, A):
        z_hat = A @ z.view(z.size(0), self.D, 1)
        AtAz = (A.T @ z_hat).squeeze(2)
        term1 = torch.sum(
            torch.norm(z - AtAz, p=2)
        )

        term2 = torch.norm(
            A @ A.T - self.Id, p=2
        ) ** 2

        return self.lambda1 * term1 + self.lambda2 * term2


class RSRAutoEncoder(nn.Module):
    def __init__(self, input_dim, d, D):
        super().__init__()
        # Put your encoder network here, remember about the output D-dimension
        self.encoder = nn.Sequential(
          nn.Linear(input_dim, input_dim // 2),  # (768, 384)
          nn.LeakyReLU(),
          nn.Linear(input_dim // 2, input_dim // 4),  # (384, 192)
          nn.LeakyReLU(),
          nn.Linear(input_dim // 4, D) # (192, 128) #? (96)
        )
        self.rsr = RSRLayer(d, D)

        # Put your decoder network here, rembember about the input d-dimension
        self.decoder = nn.Sequential(
          nn.Linear(d, D),
          nn.LeakyReLU(),
          nn.Linear(D, input_dim // 2),
          nn.LeakyReLU(),
          nn.Linear(input_dim // 2, input_dim)
        )
    
    def forward(self, x):
        enc = self.encoder(x) # obtain the embedding from the encoder
        latent = self.rsr(enc) # RSR manifold
        dec = self.decoder(latent) # obtain the representation in the input space
        return enc, dec, latent, self.rsr.A   # latent.shape [batch_size, d, 1]

class L2p_Loss(nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p
  
    def forward(self, y_hat, y):
        return torch.sum(
            torch.pow(
                torch.norm(y - y_hat, p=2), self.p
            )
        ) / (y.shape[0] * y.shape[1])

class RSRBertModel(nn.Module):
    def __init__(self, pretrained_path, bert_config, args, num_labels):
        super(RSRBertModel, self).__init__()
        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.d = args.intrinsic_dim
        self.D = args.encoder_dim
        self.ae = RSRAutoEncoder(bert_config.hidden_size, self.d, self.D)
        self.reconstruction_loss = L2p_Loss(p=args.p)
        self.rsr_loss = RSRLoss(args.lamda1, args.lamda2, self.d, self.D)
        self.classifier = nn.Linear(args.intrinsic_dim, num_labels)
        self.args = args


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        
        enc, dec, latent, rsr = self.ae(pooled_output)
        logits = self.classifier(latent)
        
        rec_loss = self.reconstruction_loss(dec, pooled_output)
        rsr_loss = self.rsr_loss(enc, rsr)
        # print(f"rec_loss: {rec_loss}  rsr_loss: {rsr_loss}")
        if self.args.reloss:
            loss = rec_loss + rsr_loss
        else:
            loss = rsr_loss
        return logits, loss


if __name__ == "__main__":
    W = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(16, 128)))
    print(W.shape)


