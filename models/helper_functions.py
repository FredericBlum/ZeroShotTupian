from conllu import parse
from torch.utils.data import random_split
from flair.data import Corpus, Dictionary, Sentence
from flair.datasets import ColumnCorpus


def conllu_to_flair(path_in):
    data = []
    word_dict = Dictionary()
    gold_dict = {}

    with open(path_in) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:                           
            utterance = []
            
            for tok in line:
                if tok['upos'] != "_":
                    tok['form'] = tok['form'].replace("-", "")
                    combined = tok['form'] + " " + tok['upos'] + " " + str(tok['head']) + " " + tok['deprel']
                    utterance.append(combined)

                    if tok['form'] in gold_dict:
                        if tok['deprel'] != gold_dict[tok['form']]:
                            print(f"you have a problem. check '{tok['form']}' manually")
                    else:
                        gold_dict[tok['form']] = tok['deprel']
                    word_dict.add_item(tok['form'])

            utt_str = "\n".join(utterance)
            data.append(utt_str)

    dev, test, train = random_split(data, [65, 65, 534])
    columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}
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

    return corpus, gold_dict, word_dict