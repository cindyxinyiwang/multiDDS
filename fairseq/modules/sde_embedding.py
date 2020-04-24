import torch
import torch.nn as nn

from .layer_norm import LayerNorm


class SDE(nn.Module):
    def __init__(self, dictionary, ngram_pool_mode='sum', n=4, threshold=32000, dim=128, latent=10000,
                 do_layer_norm=False):
        super(SDE, self).__init__()
        self.padding_idx = dictionary.pad_index
        self.embedding_dim = dim
        word_vocab = dictionary.symbols[dictionary.nspecial:]
        word_ngrams = [self.to_ngram(w, n=n) for w in word_vocab]

        from collections import Counter
        all_ngrams = Counter(sum(word_ngrams, []))
        top_ngrams = all_ngrams.most_common(threshold)
        print(f'BUILDING SDE LAYER, using n={n}, TOTAL ngram {len(all_ngrams)}, suppressing to {len(top_ngrams)}')
        print(f'ngram cutoff at {top_ngrams[-1][1]}')
        top_ngrams_symbols = [x[0] for x in top_ngrams]
        ngram_to_id = {s: i + 1 for i, s in enumerate(top_ngrams_symbols)}
        UNK_ID = 0
        ngram_to_id['<UNK>'] = UNK_ID

        ngrams_id = [[ngram_to_id.get(w, UNK_ID) for w in ww] for ww in word_ngrams]
        ngram_offsets = [0]
        for xx in ngrams_id[1:]:
            ngram_offsets.append(ngram_offsets[-1] + len(xx))
        self.register_buffer('ngram_offsets', torch.LongTensor(ngram_offsets))
        self.register_buffer('ngram_ids', torch.LongTensor(sum(ngrams_id, [])))

        self.ngram_emb = nn.EmbeddingBag(len(ngram_to_id), embedding_dim=dim, mode=ngram_pool_mode)
        self.latent_mat = nn.Parameter(torch.empty(latent, dim))
        self.special_emb = nn.Parameter(torch.empty(dictionary.nspecial, dim))

        # run weight initialization
        nn.init.normal_(self.ngram_emb.weight, mean=0, std=dim ** -0.5)
        nn.init.normal_(self.latent_mat, mean=0, std=latent ** -0.5)
        nn.init.normal_(self.special_emb, mean=0, std=dim ** -0.5)
        nn.init.constant_(self.special_emb[self.padding_idx], 0.0)

        if do_layer_norm:
            self.layer_norm = LayerNorm(dim)
        else:
            self.layer_norm = None

        self.sde_weight = None

    @staticmethod
    def to_ngram(word: str, n=4):
        ngrams = []
        for l in range(1, min(n + 1, len(word) + 1)):
            for start, end in zip(range(len(word)), range(l, len(word) + 1)):
                ngrams.append(word[start:end])
        return ngrams

    def forward(self, x):
        # build current iteration weight matrix
        # BOW
        ngram_weight = self.ngram_emb(self.ngram_ids, self.ngram_offsets)  # N_ng * dim
        ngram_weight = torch.tanh(ngram_weight)
        # latent
        latent_scores = torch.matmul(ngram_weight, self.latent_mat.transpose(0, 1))
        latent_distribution = torch.softmax(latent_scores, dim=-1)
        latent_emb = torch.matmul(latent_distribution, self.latent_mat)
        # residual connection
        token_emb = ngram_weight + latent_emb  # threshold * dim

        emb_weight = torch.cat((self.special_emb, token_emb), dim=0)
        self.sde_weight = emb_weight
        sde_emb = nn.functional.embedding(x, emb_weight, padding_idx=self.padding_idx)
        if self.layer_norm:
            sde_emb = self.layer_norm(sde_emb)
        return sde_emb

    @property
    def weight(self):
        return self.sde_weight


class SDENoWeight(SDE):
    def forward(self, x, lang):
        # construct mask
        special_mask = (x < len(self.special_emb)).unsqueeze(-1)  # B * L
        non_special_mask = ~special_mask  # B * L
        # build current iteration weight matrix
        # BOW
        ngram_weight = self.ngram_emb(self.ngram_ids, self.ngram_offsets) # N_ng * dim
        ngram_weight = torch.tanh(ngram_weight)
        ngram_emb_weight = torch.cat((self.special_emb, ngram_weight), dim=0) # len(dict) * dim
        lang_indep_emb = nn.functional.embedding(x, ngram_emb_weight, padding_idx=self.padding_idx)  # B * L * dim
        # lang specific
        lang_emb = self.language_transformations[lang](lang_indep_emb)  # B * L * dim
        lang_emb = torch.tanh(lang_emb)
        # latent
        latent_scores = torch.matmul(lang_emb, self.latent_mat.transpose(0, 1))
        latent_distribution = torch.softmax(latent_scores, dim=-1)
        latent_emb = torch.matmul(latent_distribution, self.latent_mat)
        # residual connection
        final_word_emb = lang_emb + latent_emb
        # piece up things
        final_word_emb = non_special_mask.type(final_word_emb.dtype) * final_word_emb
        special_emb = special_mask.type(lang_indep_emb.dtype) * lang_indep_emb
        final_emb = final_word_emb + special_emb
        return final_emb

    @property
    def weight(self):
        raise NotImplementedError(f'{type(self).__name__} does not support getting full weight')
