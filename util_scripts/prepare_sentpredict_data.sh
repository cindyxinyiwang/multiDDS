
python preprocess.py \
	--only-source \
	--trainpref "data/dialects/fra/prediction16k/train.input0" \
	--validpref "data/dialects/fra/prediction16k/dev.input0" \
	--testpref "data/dialects/fra/prediction16k/test.input0" \
	--destdir "data-bin/dialect_fra_predict16k/input0" \
	--workers 32 \
	--srcdict "data-bin/dialects_fra16k/dict.fra.txt" 

python preprocess.py \
	--only-source \
	--trainpref "data/dialects/fra/prediction16k/train.label" \
	--validpref "data/dialects/fra/prediction16k/dev.label" \
	--testpref "data/dialects/fra/prediction16k/test.label" \
	--destdir "data-bin/dialect_fra_predict16k/label" 

