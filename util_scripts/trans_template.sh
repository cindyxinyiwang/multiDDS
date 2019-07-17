
IL=
DATAPATH=
MODEL=checkpoints/debug/checkpoint_last.pt
MODELNAME=

mkidr -p trans_results/"$IL"
python interactive.py $DATAPATH \
       	--path $MODEL \
	--batch-size 1 \
	--buffer-size 2 \
	--remove-bpe sentencepiece \
	--beam 5  < test_data/ori_eng/setE-mono-standard.spm8000.dnt0.ori  > ./trans_results/"$IL"/"$MODELNAME".test.log

grep ^H ./trans_results/"$IL"/"$MODELNAME".test.log | cut -f3 > ./trans_results/"$IL"/"$MODELNAME".test.decode


