# Target Conditioned Sampling: Optimizing Data Selection for Multilingual Neural Machine Translation (Xinyi Wang, Graham Neubig, 2019)

This page includes usage of the multilingual data filtering method from the paper [Target Conditioned Sampling: Optimizing Data Selection for Multilingual Neural Machine Translation](https://arxiv.org/abs/1905.08212).


## Example usage
The directory test_data/ contains the toy training data from three languages: aze, bel, tur. To calculate the language distance of aze from each of the three languages {aze,bel,rus}:
```
python get_language_distance.py \
    --language-names "aze,bel,tur" \
    --base-language-name "aze" \
    --file-pattern "test_data/train.CODE-eng.CODE" \
    --file-pattern-src "test_data/train.CODE-eng.CODE" \
    --file-pattern-trg "test_data/train.CODE-eng.eng" 
```
To filter training data for using the three languages to improve the performance of aze:
```
python get_language_distance.py \
    --language-names "aze,bel,tur" \
    --base-language-name "aze" \
    --file-pattern "test_data/train.CODE-eng.CODE" \
    --filter-data \
    --file-pattern-src "test_data/train.CODE-eng.CODE" \
    --file-pattern-trg "test_data/train.CODE-eng.eng" 
```
