
MDIR=/projects/tir1/users/xinyiw1/usr/local/mosesdecoder

python util_scripts/post_trans_fairseq.py $1

$MDIR/scripts/tokenizer/detokenizer.perl < $1.ref > $1.ref.detok
$MDIR/scripts/tokenizer/detokenizer.perl < $1.hyp > $1.hyp.detok

cat $1.hyp.detok | sacrebleu -w 2 $1.ref.detok | tee $1.sacrebleu


