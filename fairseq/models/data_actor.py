import torch


class BaseActor(torch.nn.Module):
  def __init__(self, args, lan_size):
    super(BaseActor, self).__init__()
    self.args = args
    self.lan_size = lan_size
    # init
    self.bias = torch.nn.Linear(lan_size, 1)
    for p in self.bias.parameters():
      p.data.fill_(0.)

  def forward(self, feature):
    # input feature are 1 vector
    # feature: [1, lan_size]
    logits = self.bias.weight * feature
    return logits

class LanguageActor(torch.nn.Module):
  def __init__(self, args, lan_size):
    super(LanguageActor, self).__init__()
    self.args = args
    self.lan_size = lan_size
    embed_dim = args.data_actor_embed_dim
    self.lan_emb = Embedding(self.lan_size, embed_dim, None)
    # init
    self.w = torch.nn.Linear(embed_dim, embed_dim)
    self.project_out = torch.nn.Linear(embed_dim, 1)
    for p in self.w.parameters():
        torch.nn.init.uniform_(p, -0.1, 0.1)
    for p in self.project_out.parameters():
        torch.nn.init.uniform_(p, -0.1, 0.1)

  def forward(self, feature):
    # input feature is lan id
    # feature: [1, 1]
    emb = self.lan_emb(feature)
    x = self.w(emb)
    logits = self.project_out(emb).squeeze(2)
    return logits


class AveEmbActor(torch.nn.Module):
    """Average Embedding actor"""
    def __init__(self, args, task, emb=None, optimize_emb=True):
        super(AveEmbActor, self).__init__()
        #assert task.source_dictionary == task.target_dictionary
        src_dictionary = task.source_dictionary
        trg_dictionary = task.target_dictionary
        self.padding_idx = src_dictionary.pad()
        if emb is None:
            embed_dim = args.data_actor_embed_dim
            if src_dictionary == trg_dictionary:
                num_embeddings = len(src_dictionary)
                self.src_embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
                self.trg_embed_tokens = self.src_embed_tokens
            else:
                num_embeddings = len(src_dictionary)
                self.src_embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
                num_embeddings = len(trg_dictionary)
                self.trg_embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.src_embed_tokens = emb
            self.trg_embed_tokens = emb
            embed_dim = emb.weight.size(1)
        self.project_out = torch.nn.Linear(2*embed_dim, 1)
        self.out_score_type = args.out_score_type
        
    def forward(self, src_tokens, trg_tokens):
        bsz, seqlen = src_tokens.size()

        src_word_count = (~src_tokens.eq(self.padding_idx)).long().sum(dim=-1, keepdim=True)
        # embed tokens
        x = self.src_embed_tokens(src_tokens)
        #x = F.dropout(x, p=self.dropout_in, training=self.training)
        
        # B x T x C -> B x C
        x = x.sum(dim=1) / src_word_count.float()

        trg_word_count = (~trg_tokens.eq(self.padding_idx)).long().sum(dim=-1, keepdim=True)
        # embed tokens
        y = self.trg_embed_tokens(trg_tokens)
        
        #y = torch.nn.functional.dropout(y, p=self.dropout_in, training=self.training)
        # B x T x C -> B x C
        y = y.sum(dim=1) / trg_word_count.float()

        inp = torch.cat([x, y], dim=-1)
        # B x 1
        if self.out_score_type == 'sigmoid':
            score = torch.sigmoid(self.project_out(inp))
        elif self.out_score_type == 'exp':
            score = torch.exp(self.project_out(inp))
        return score 

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
    if padding_idx is not None:
        torch.nn.init.constant_(m.weight[padding_idx], 0)
    return m


