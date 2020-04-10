# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the # LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy

from fairseq import utils
from fairseq.models import (
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)


@register_model('multilingual_transformer')
class MultilingualTransformerModel(FairseqMultiModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders, args):
        super().__init__(encoders, decoders, args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')
        parser.add_argument('--share-all-langpair-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
 
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024
        if not hasattr(args, 'share_all_langpair_embeddings'):
            args.share_all_langpair_embeddings = False

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        elif args.share_all_langpair_embeddings:
            lang_pair_embed = {}
            for lang_pair in task.model_lang_pairs:
                print(lang_pair)
                lang_pair_embed[lang_pair] = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=task.langs,
                    embed_dim=args.encoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.encoder_embed_path,
                )
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=src_langs,
                        embed_dim=args.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.encoder_embed_path,
                    )
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=tgt_langs,
                        embed_dim=args.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.decoder_embed_path,
                    )
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang, lang_pair=None):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                elif args.share_all_langpair_embeddings:
                    encoder_embed_tokens = lang_pair_embed[lang_pair]
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path
                    )
                if hasattr(args, "reset_bt_dropout") and args.reset_bt_dropout:
                    copied_args = copy.deepcopy(args)
                    if lang_pair == args.bt_langpair:
                        copied_args.dropout = args.bt_dropout
                        copied_args.attention_dropout = args.bt_attention_dropout
                        copied_args.relu_dropout = args.bt_relu_dropout
                    lang_encoders[lang] = TransformerEncoder(copied_args, task.dicts[lang], encoder_embed_tokens)
                else:
                    lang_encoders[lang] = TransformerEncoder(args, task.dicts[lang], encoder_embed_tokens)
            return lang_encoders[lang]

        def get_decoder(lang, lang_pair=None):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                elif args.share_all_langpair_embeddings:
                    decoder_embed_tokens = lang_pair_embed[lang_pair]
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )
                if hasattr(args, "reset_bt_dropout") and args.reset_bt_dropout:
                    copied_args = copy.deepcopy(args)
                    if lang_pair == args.bt_langpair:
                        copied_args.dropout = args.bt_dropout
                        copied_args.attention_dropout = args.bt_attention_dropout
                        copied_args.relu_dropout = args.bt_relu_dropout
                    lang_decoders[lang] = TransformerDecoder(copied_args, task.dicts[lang], decoder_embed_tokens)
                else:
                    lang_decoders[lang] = TransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.encoder_lang_group:
            shared_encoder_group = {}
            for lang_group in args.encoder_lang_group:
                shared_encoder_group[lang_group[0]] = get_encoder(lang_group[0])
                for l in lang_group[1:]:
                    shared_encoder_group[l] = shared_encoder_group[lang_group[0]]
        elif args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.decoder_lang_group:
            shared_decoder_group = {}
            for lang_group in args.decoder_lang_group:
                shared_decoder_group[lang_group[0]] = get_decoder(lang_group[0])
                for l in lang_group[1:]:
                    shared_decoder_group[l] = shared_decoder_group[lang_group[0]]    
        elif args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])            

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            if args.encoder_lang_group:
                encoders[lang_pair] = shared_encoder_group[src]
            else:
                encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src, lang_pair)
            if args.decoder_lang_group:
                decoders[lang_pair] = shared_decoder_group[tgt]
            else:
                decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt, lang_pair)

        ret = MultilingualTransformerModel(encoders, decoders, args)
        ret.embed_tokens = shared_encoder_embed_tokens
        return ret 

    def load_state_dict(self, state_dict, strict=True):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            #assert k.startswith('models.')
            lang_pair = k.split('.')[1]
            if lang_pair == "eng-ces":
                name_toks = k.split('.')
                name_toks[1] = "eng-slk"
                state_dict_subset[".".join(name_toks)] = state_dict_subset[k]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=False)

        #for key, m in self.models.items():
        #    m.load_state_dict(state_dict_subset)


@register_model_architecture('multilingual_transformer', 'multilingual_transformer')
def base_multilingual_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)

@register_model_architecture('multilingual_transformer', 'multilingual_wmt_transformer')
def multilingual_wmt_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.2)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.2)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.2)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

@register_model_architecture('multilingual_transformer', 'multilingual_transformer_iwslt_de_en')
def multilingual_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    #args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    #args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    #args.dropout = getattr(args, 'dropout', 0.)

    base_multilingual_architecture(args)
