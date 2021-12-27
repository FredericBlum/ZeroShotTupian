from helper_functions import create_sk_text
from flair.datasets import ColumnCorpus
from flair.data import Corpus


create_sk_text('../data/shipibo/orig/train.conllu', '../data/shipibo/flair/train.txt', sep = False)
create_sk_text('../data/shipibo/orig/valid.conllu', '../data/shipibo/flair/valid.txt', sep = False)
create_sk_text('../data/shipibo/orig/test.conllu', '../data/shipibo/flair/test.txt', sep = False)

# init a corpus using column format, data folder and the names of the train, dev and test files
columns = {0: 'text', 1: 'upos', 2: 'deprel'}
data_folder = '../data/shipibo/flair'

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'train.txt',
                              test_file = 'test.txt',
                              dev_file = 'valid.txt')

print(corpus.train[0])

# create label dictionary for a Universal Part-of-Speech tagging task
upos_dictionary = corpus.make_label_dictionary(label_type='upos')
deprel_dictionary = corpus.make_label_dictionary(label_type='deprel')

from typing import Dict

def gold_dictionary(data, unk_threshold: int = 0) -> Dict[str, str]:
    '''
    Makes a dictionary of words and their deprel, given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and label values
    '''

    # First count the frequency of each distinct ngram
    gold_dict = {}
    for sent in data:
        print(sent)
        for word, label in sent:
            if word not in gold_dict:
                gold_dict[word] = label
            else:
                print(f"problem with word '{word}' and label '{label}'")
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix

"""     # Assign indices to each distinct ngram
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix) """

    # Print some info on dictionary size


gold_dictionary(corpus.train)
