

MODEL=pretrained_models/xlmr.large/sentencepiece.bpe.model
DICT=pretrained_models/xlmr.large/dict.txt

#for file in xnli/data/*.input0*; do
#  f=${file/input0/input0.spm}
#  spm_encode --model=$MODEL --output_format=piece < $file > $f
#done
#
#for file in xnli/data/*.input1*; do
#  f=${file/input1/input1.spm}
#  spm_encode --model=$MODEL --output_format=piece < $file > $f
#done
#
#for file in xnli/data/*.label*; do
#  f=${file/label/label.spm}
#  spm_encode --model=$MODEL --output_format=piece < $file > $f
#done

#for LAN in ar bg de el en es fr hi ru sw th tr ur vi zh; do
#  for type in input0 input1; do
#    python preprocess.py \
#      --only-source \
#      --trainpref xnli/data/xnli.train."$type".spm."$LAN" \
#      --validpref xnli/data/xnli.dev."$type".spm."$LAN" \
#      --testpref xnli/data/xnli.test."$type".spm."$LAN" \
#      --destdir data-bin/xnli/"$LAN"/"$type" \
#      --workers 32 \
#      --srcdict $DICT 
#  done
#done

python preprocess.py \
  --only-source \
  --trainpref xnli/data/xnli.train.label.ar \
  --validpref xnli/data/xnli.dev.label.ar \
  --testpref xnli/data/xnli.test.label.ar \
  --destdir data-bin/xnli/ar/label \
  --workers 32

for LAN in  bg de el en es fr hi ru sw th tr ur vi zh; do
   python preprocess.py \
     --only-source \
     --trainpref xnli/data/xnli.train.label."$LAN" \
     --validpref xnli/data/xnli.dev.label."$LAN" \
     --testpref xnli/data/xnli.test.label."$LAN" \
     --destdir data-bin/xnli/"$LAN"/label \
     --srcdict data-bin/xnli/ar/label/dict.txt \
     --workers 32 
done


