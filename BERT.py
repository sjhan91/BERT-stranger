import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from einops import rearrange
from utils.vocab import *
from rotary_embedding_torch import RotaryEmbedding

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace


class Embedding(nn.Module):
    def __init__(self, dim, num_vocab, max_len):
        super().__init__()

        self.token_embed = nn.Embedding(num_vocab, dim)
        self.pos_embed = FreqEmbedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        embed = self.token_embed(x)

        pos = self.pos_embed(x)
        pos = pos.type_as(embed)

        embed = self.norm(embed + pos)

        return embed


class FreqEmbedding(nn.Module):
    """
    Refer to https://github.com/dreamgonfly/transformer-pytorch/blob/master/embeddings.py
    """

    def __init__(self, max_len, dim):
        super().__init__()

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, dim)
        pe.require_grad = False

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = (torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, : x.shape[1]]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    """
    Refer to https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    """

    def __init__(self, dim, heads=12, dim_head=64):
        super().__init__()

        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = dim_head * heads

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(dim=32)

    def forward(self, x, mask=None):
        # x -> (batch (b), seq (n), dim (d))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # rotary embedding
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = mask.unsqueeze(1)

        if mask is not None:
            fill_value = 1e-9 if dots.dtype == torch.float32 else 1e-4
            dots.masked_fill_(mask, fill_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    """
    Refer to https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        self.dropout = nn.Dropout(rate)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = self.norm1(self.dropout(attn(x, mask)) + x)
            x = self.norm2(self.dropout(ff(x)) + x)

        return x


class BERT(nn.Module):
    def __init__(self, dim, vocab, bpe_vocab, depth, heads, dim_head, mlp_dim, max_len, rate):
        super().__init__()

        self.vocab = vocab
        self.bpe_vocab = bpe_vocab

        num_vocab = bpe_vocab.get_vocab_size()
        self.embedding = Embedding(dim, num_vocab, max_len)

        self.transformer = Transformer(
            dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            rate=rate,
        )

        self.linear_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_vocab),
        )

    def get_attn_pad_mask(self, seq):
        batch_size, len_seq = seq.shape

        pad_idx = self.vocab.to_i(PAD_TOKEN)
        pad_attn_mask = seq.eq(pad_idx).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, len_seq, len_seq)

        return pad_attn_mask

    def forward(self, x):
        attn_mask = self.get_attn_pad_mask(x)

        # Transformers
        x = self.embedding(x)
        h = self.transformer(x, attn_mask)

        # last layer
        logits = self.linear_head(h)

        return logits, h[:, 0]


class BERT_Lightning(pl.LightningModule):
    # BERT base: L=12, H=768, A=12
    def __init__(
        self,
        dim,
        depth=12,
        heads=12,
        dim_head=64,
        mlp_dim=2048,
        max_len=512,
        rate=0.1,
        loss_weights=[1, 1],
        lr=1e-3,
        warm_up=5000,
        bpe_path="",
    ):
        super().__init__()

        bpe_vocab = Tokenizer(BPE())
        bpe_vocab.pre_tokenizer = Whitespace()
        bpe_vocab = bpe_vocab.from_file(bpe_path)

        self.vocab = RemiVocab()
        self.bpe_vocab = bpe_vocab

        self.model = BERT(
            dim=dim,
            vocab=self.vocab,
            bpe_vocab=self.bpe_vocab,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            max_len=max_len,
            rate=rate,
        )

        self.lr = lr
        self.warm_up = warm_up

        self.loss_weights = loss_weights
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))

    def forward(self, x):
        logits, h = self.model(x)

        return logits, h

    def configure_optimizers(self):
        """
        Refer to https://gist.github.com/gautierdag/925760d4295080c1860259dba43e4c01
        """

        opt = AdamW(self.parameters(), lr=self.lr)

        def warm_decay(step):
            if step < self.warm_up:
                return step / self.warm_up
            return self.warm_up**0.5 * step**-0.5

        sch = {
            "scheduler": LambdaLR(opt, warm_decay),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }

        return [opt], [sch]

    def get_acc(self, y_pred, y_true):
        y_pred = nn.Softmax(dim=-1)(y_pred)
        y_pred = y_pred.argmax(-1)

        # exclude pad token for the metric
        nonzero_idx = y_true != self.vocab.to_i(PAD_TOKEN)

        numer = (y_pred[nonzero_idx] == y_true[nonzero_idx]).sum()
        denom = y_pred[nonzero_idx].numel()

        return numer / denom

    def compute_loss(self, batch):
        y_mask_pred, h_mask_pred = self.model(batch["x_mask"])
        y_pos_pred, h_pos_pred = self.model(batch["x_neigh"])

        # MLM loss
        mlm_loss = self.ce_loss(y_mask_pred.transpose(1, 2), batch["y_mask"])
        mlm_acc = self.get_acc(y_mask_pred, batch["y_mask"])

        # contrastive loss
        rcl_loss = self.compute_rcl_loss(h_mask_pred, h_pos_pred)

        return mlm_loss, mlm_acc, rcl_loss

    def compute_rcl_loss(self, anchor, positive):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)

        rcl_loss = (anchor - positive).pow(2).sum(1).mean()
        rcl_loss = -rcl_loss

        return rcl_loss

    def training_step(self, train_batch, batch_idx):
        mlm_loss, mlm_acc, rcl_loss = self.compute_loss(train_batch)

        # total loss
        loss = (self.loss_weights[0] * mlm_loss) + (self.loss_weights[1] * rcl_loss)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mlm_loss", mlm_loss, prog_bar=True)
        self.log("train_mlm_acc", mlm_acc, prog_bar=True)
        self.log("train_rcl_loss", rcl_loss, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        mlm_loss, mlm_acc, rcl_loss = self.compute_loss(val_batch)

        # total loss
        loss = (self.loss_weights[0] * mlm_loss) + (self.loss_weights[1] * rcl_loss)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mlm_loss", mlm_loss, prog_bar=True, sync_dist=True)
        self.log("val_mlm_acc", mlm_acc, prog_bar=True, sync_dist=True)
        self.log("val_rcl_loss", rcl_loss, prog_bar=True, sync_dist=True)

        return loss
