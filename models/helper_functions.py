from conllu import parse
from torch.utils.data import random_split
from flair.data import Corpus, Dictionary
from flair.datasets import ColumnCorpus


def conllu_to_flair(path_in, lang):
    data = []
    count: int = 0

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
                    count += 1

            utt_str = "\n".join(utterance)
            data.append(utt_str)

    overall = len(data)
    tenth = int(round((overall/10), 0))
    rest = int(overall - tenth - tenth)
    dev, test, train = random_split(data, [tenth, tenth, rest], generator=torch.Generator().manual_seed(1))
    columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}
    data_folder = f'./data/{lang}/flair'

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
    
    print(f"The {lang} corpus contains {count} tokens in total.")
    print(corpus)

    return corpus

def make_dictionary(path_in, unk_threshold: int = 0):
    with open(path_in) as file:
        doc = file.read()

    word_frequencies = {}
    data = doc.split(" ")
    for word in data:
        if word not in word_frequencies:
            word_frequencies[word] = 0
        word_frequencies[word] += 1

    word_dict = Dictionary()
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:
            word_dict.add_item(word)

    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_dict)} words")
    return word_dict