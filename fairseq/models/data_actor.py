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

class AveEmbActor(torch.nn.Module):
    """Average Embedding actor"""
    def __init__(self, args, task, emb=None, optimize_emb=True):
        super(AveEmbActor, self).__init__()
        assert task.source_dictionary == task.target_dictionary
        dictionary = task.source_dictionary
        self.padding_idx = dictionary.pad()
        if emb is None:
            embed_dim = args.data_actor_embed_dim
            num_embeddings = len(dictionary)
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = emb
            embed_dim = emb.weight.size(1)
        self.project_out = torch.nn.Linear(2*embed_dim, 1)
        self.out_score_type = args.out_score_type
        
    def forward(self, src_tokens, trg_tokens):
        bsz, seqlen = src_tokens.size()

        src_word_count = (~src_tokens.eq(self.padding_idx)).long().sum(dim=-1, keepdim=True)
        # embed tokens
        x = self.embed_tokens(src_tokens)
        #x = F.dropout(x, p=self.dropout_in, training=self.training)
        
        # B x T x C -> B x C
        x = x.sum(dim=1) / src_word_count.float()

        trg_word_count = (~trg_tokens.eq(self.padding_idx)).long().sum(dim=-1, keepdim=True)
        # embed tokens
        y = self.embed_tokens(trg_tokens)
        
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

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
    torch.nn.init.constant_(m.weight[padding_idx], 0)
    return m


