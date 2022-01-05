# LangDocNLP

code for supar biaffine dependency parser:

```python
python -u -m supar.cmds.biaffine_dependency train -b -d 0 -c config-ini -p guacamole -f char --embed ../glove/vectors.txt --train data/Guajajara/conllu/train.conllu --dev data/Guajajara/conllu/dev.conllu --test data/Guajajara/conllu/test.conllu --bert bert-base-multilingual-cased --n-embed 512 --unk=''
```

Code for evaluation:

```python
python -u -m supar.cmds.biaffine_dependency evaluate -d 0 -p guacamole/model --data data/Guajajara/conllu/all_in_one.conllu --tree  --proj
```
