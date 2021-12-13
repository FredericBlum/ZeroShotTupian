from typing import Dict
from conllu import parse
import torch
from more_itertools import windowed

def read_conllu(path, sep = False, char = False):
    data = []
    with open(path) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:
            for tok in line:
                if tok['upos'] != "_":

                    if char == True:
                        char_list = []

                        for char in tok['form']:
                            if sep == True:
                                char = char.replace("-", "")
                                char_list.append(char)
                            else:
                                char_list.append(char)

                        data.append((char_list, tok['upos']))
                    else:
                        data.append((tok['form'], tok['upos']))
                        
                    # print(char_list, tok['upos'])
    return data

def read_conllu_utt(path):
    data = []
    with open(path) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:
            utterance = []
            for tok in line:
                if tok['upos'] != "_":
                    utterance.append(tok['form'])

            utt_str = " ".join(utterance)
            data.append(utt_str)
    print(data)

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


def make_label_dictionary(data) -> Dict[str, int]:
    '''
    Make a dictionary of labels.
    :param data: List of (sentence, label) tuples
    :return: A dictionary of string keys and index values
    '''
    label_to_ix = {}
    for _, label in data:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
    return label_to_ix

def make_label_dictionary(data) -> Dict[str, int]:
    '''
    Make a dictionary of labels.
    :param data: List of (sentence, label) tuples
    :return: A dictionary of string keys and index values
    '''
    label_to_ix = {}
    label_freq = {}
    for _, label in data:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
            label_freq[label] = 1
        else:
            label_freq[label] += 1

    return label_to_ix, label_freq


def make_label_vector(label, label_to_ix):
    device = 'cuda'
    return torch.LongTensor([label_to_ix[label]], device=device)


def make_onehot_vectors(sentence, word_to_ix, max_ngrams: int = 1):
    device = 'cuda'
    onehot_vectors = []

    # go over all n-gram sizes (including 1)
    for ngram_size in range(1, max_ngrams + 1):

        # move a window over the text
        for ngram in windowed(sentence, ngram_size):
            if None not in ngram:

                # make ngram string
                ngram_word = " ".join(ngram)

                # look up ngram index in dictionary
                if ngram_word in word_to_ix:
                    onehot_vectors.append(word_to_ix[ngram_word])
                else:
                    onehot_vectors.append(word_to_ix["UNK"] if "UNK" in word_to_ix else 0)

    return torch.tensor(onehot_vectors, devive=device).unsqueeze(0)