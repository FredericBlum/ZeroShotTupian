from typing import Dict
from conllu import parse
from torch.utils.data import random_split
from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus


def make_dictionary(data, unk_threshold: int = 0) -> Dict[str, int]:
    # First count the frequency of each distinct ngram
    for sent in data:
        for word in sent:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1

    # Assign indices to each distinct ngram
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix)

    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix

def conllu_to_flair(path_in):
    data = []
    word_dict = {}
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

                    if tok['form'] in gold_dict:
                        if tok['deprel'] != gold_dict[tok['form']]:
                            print(f"you have a problem. check '{tok['form']}' manually")
                    else:
                        gold_dict[tok['form']] = tok['deprel']
                        word_dict[tok['form']] = len(word_dict)
                    utterance.append(combined)

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