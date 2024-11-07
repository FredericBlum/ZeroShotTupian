# Evaluating zero-shot transfers and multilingual models for dependency parsing and POS tagging within the low-resource language family Tupían

This repository documents the code used in the following paper:

> Frederic Blum. 2022. Evaluating zero-shot transfers and multilingual models for dependency parsing and POS tagging within the low-resource language family Tupían. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages 1–9, Dublin, Ireland. Association for Computational Linguistics. <https://doi.org/10.18653/v1/2022.acl-srw.1>

## Some notes on the code

code for supar biaffine dependency parser:

```python
python -u -m supar.cmds.biaffine_dependency train -b -d 0 -c config-ini -p guacamole -f char --embed ../glove/vectors.txt --train data/Guajajara/conllu/train.conllu --dev data/Guajajara/conllu/dev.conllu --test data/Guajajara/conllu/test.conllu --bert bert-base-multilingual-cased --n-embed 512 --unk=''
```

Code for evaluation:

```python
python -u -m supar.cmds.biaffine_dependency evaluate -d 0 -p guacamole/model --data data/Guajajara/conllu/all_in_one.conllu --tree  --proj
```
