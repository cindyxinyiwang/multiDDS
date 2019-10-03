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
    def __init__(self, args, task):
        super(AveEmbActor, self).__init__()
        assert task.source_dictionary == task.target_dictionary
        dictionary = task.source_dictionary
        embed_dim = args.encoder_embed_dim
        hidden_size=args.encoder_hidden_size
        dropout_in=args.encoder_dropout_in
        dropout_out=args.encoder_dropout_out

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)

        self.padding_value = padding_value
        self.output_units = hidden_size

        self.project_out = nn.Linear(2*embed_dim, 1)

    def forward(self, src_tokens, trg_tokens):
        bsz, seqlen = src_tokens.size()

        src_word_count = src_tokens.eq(self.padding_idx).long().sum(dim=-1, keepdim=True)
        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        # B x T x C -> B x C
        x = x.sum(dim=1) / src_word_count

        trg_word_count = trg_tokens.eq(self.padding_idx).long().sum(dim=-1, keepdim=True)
        # embed tokens
        y = self.embed_tokens(trg_tokens)
        y = F.dropout(y, p=self.dropout_in, training=self.training)
        # B x T x C -> B x C
        y = y.sum(dim=1) / trg_word_count

        inp = torch.cat([x, y], dim=-1)
        # B x 1
        score = self.project_out(inp)
        return score 

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


