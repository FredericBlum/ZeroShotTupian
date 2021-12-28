from typing import Dict
from conllu import parse
from torch.utils.data import random_split
from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus

def read_conllu(path, sep: bool = True, char: bool = False):
    data = []
    with open(path) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:
            for tok in line:
                if tok['upos'] != "_":
                    if sep == False:
                        tok['form'] = tok['form'].replace("-", "")

                    if char == True:
                        char_list = []

                        for char in tok['form']:
                            char_list.append(char)

                        data.append((char_list, tok['upos']))
                    else:
                        data.append((tok['form'], tok['upos']))

    return data

def conllu_to_flair(path_in):
    data = []
    gold_dict = {}

    with open(path_in) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:                           
            utterance = []
            
            for tok in line:
                if tok['upos'] != "_":
                    tok['form'] = tok['form'].replace("-", "")
                    combined = tok['form'] + " " + tok['upos'] + " " + tok['deprel']

                    if tok['form'] in gold_dict:
                        if tok['deprel'] != gold_dict[tok['form']]:
                            print(f"you have a problem. check '{tok['form']}' manually")
                    else:
                        gold_dict[tok['form']] = tok['deprel']
                    utterance.append(combined)

            utt_str = "\n".join(utterance)
            data.append(utt_str)

    dev, test, train = random_split(data, [65, 65, 534])
    columns = {0: 'text', 1: 'upos', 2: 'deprel'}
    data_folder = './data/shipibo/flair'

    dev = "\n\n".join(dev)
    test = "\n\n".join(test)
    train = "\n\n".join(train)

    with open(f'{data_folder}/dev.txt', 'w') as f:
        f.write(dev)
    with open(f'{data_folder}/test.txt', 'w') as f:
        f.write(test)
    with open(f'{data_folder}/train.txt', 'w') as f:
        f.write(train)

    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file = 'train.txt',
                                test_file = 'test.txt',
                                dev_file = 'dev.txt')

    return corpus, gold_dict