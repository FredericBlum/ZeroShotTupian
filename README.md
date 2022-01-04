# LangDocNLP

code for supar biaffine dependency parser:

```python
python -u -m supar.cmds.biaffine_dependency train -b -d 0 -p gua/model -f char --embed ../glove/vectors.txt --train data/Guajajara/conllu/train.conllu --dev data/Guajajara/dev.conllu --test data/Guajajara/test.conllu --bert bert-base-multilingual-cased --n-embed 512 --unk=''
```

Code for evaluation:

```python
python -u -m supar.cmds.biaffine_dep evaluate -d 0 -p biaffine-dep-en --data ptb/test.conllx --tree  --proj
```
