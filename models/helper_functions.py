from typing import Dict
from conllu import parse
from torch.utils.data import random_split

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
    data_folder = '../data/shipibo/flair'

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

def read_conllu_utt(path):
    data = []
    with open(path) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:
            utterance = []
            for tok in line:
                if tok['upos'] != "_":
                    tok['form'] = tok['form'].replace("-", "")

                    utterance.append(tok['form'])

            utt_str = " ".join(utterance)
            data.append(utt_str)

    data = "\n".join(data)
    #print(data)

    return data


def make_word_dictionary(data, unk_threshold: int = 0, max_ngrams: int = 1) -> Dict[str, int]:
    '''
    Makes a dictionary of words given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and index values
    '''

    # First count the frequency of each distinct ngram
    word_frequencies = {}
    for sent, _ in data:

        # go over all n-gram sizes (including 1)
        for ngram_size in range(1, max_ngrams + 1):

            # move a window over the text
            for ngram in windowed(sent, ngram_size):

                if None not in ngram:
                    # create ngram string and count frequencies
                    ngram_word = " ".join(ngram)
                    if ngram_word not in word_frequencies:
                        word_frequencies[ngram_word] = 0
                    word_frequencies[ngram_word] += 1

    # Assign indices to each distinct ngram
    word_to_ix = {'UNK': 0}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix)

    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix