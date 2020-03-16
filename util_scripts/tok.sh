
MDIR=/projects/tir1/users/xinyiw1/usr/local/mosesdecoder

python util_scripts/post_trans_fairseq.py $1

$MDIR/scripts/tokenizer/tokenizer.perl < $1.ref > $1.ref.tok
$MDIR/scripts/tokenizer/tokenizer.perl < $1.hyp > $1.hyp.tok

./multi-bleu.perl $1.ref.tok < $1.hyp.tok
#cat $1.hyp.detok | sacrebleu -w 2 $1.ref.detok | tee $1.sacrebleu


