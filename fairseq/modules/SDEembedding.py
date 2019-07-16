
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

class QueryEmb(nn.Module):
  def __init__(self, vocab_size, d_vec):
    super(QueryEmb, self).__init__()
    self.vocab_size = vocab_size
    self.dropout = nn.Dropout(0.1)
    self.emb_matrix = nn.Parameter(torch.ones(vocab_size, d_vec).uniform_(-0.1, 0.1), requires_grad=True)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, q):
    """ 
    dot prodct attention: (q * k.T) * v
    Args:
      q: [batch_size, d_q] (target state)
      k: [len_k, d_k] (source enc key vectors)
      v: [len_v, d_v] (source encoding vectors)
      attn_mask: [batch_size, len_k] (source mask)
    Return:
      attn: [batch_size, d_v]
    """

    batch_size, max_len, d_q = q.size()
    # [batch_size, max_len, vocab_size]
    #attn_weight = torch.bmm(q, self.emb_matrix.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)) / self.temp
    attn_weight = torch.bmm(q, self.emb_matrix.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1))
    attn_weight = self.softmax(attn_weight)
    attn_weight = self.dropout(attn_weight)
    # [batch_size, max_len, d_emb_dim]
    ctx = torch.bmm(attn_weight, self.emb_matrix.unsqueeze(0).expand(batch_size, -1, -1))
    ctx = ctx + q
    return ctx

class charEmbedder(nn.Module):
  def __init__(self, char_vsize, d_vec, *args, **kwargs):
    super(charEmbedder, self).__init__()

    self.char_emb_proj = nn.Linear(char_vsize, d_vec, bias=False)

  def forward(self, x_train_char, file_idx=None):
    """Performs a forward pass.
    Args:
    Returns:
    """
    for idx, x_char_sent in enumerate(x_train_char):
      emb = x_char_sent.to_dense()
      #if self.hparams.cuda: emb = emb.cuda()
      x_char_sent = torch.tanh(self.char_emb_proj(emb))

      x_train_char[idx] = x_char_sent
    
    char_emb = torch.stack(x_train_char, dim=0)

    return char_emb


class SDEembedding(nn.Module):
  def __init__(self, char_vsize, d_vec, vocab_size=10000, padding_idx=None):
    super(SDEembedding, self).__init__()
    self.char_emb = charEmbedder(char_vsize, d_vec)
    self.query_emb = QueryEmb(vocab_size, d_vec)
    self.embedding_dim = d_vec
    self.padding_idx = padding_idx

  def forward(self, tokens):
    char_emb = self.char_emb(tokens)
    word_emb = self.query_emb(char_emb)
    return word_emb


