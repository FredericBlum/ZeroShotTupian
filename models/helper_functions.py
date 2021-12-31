from conllu import parse
from torch.utils.data import random_split
import torch
from flair.data import Corpus, Dictionary
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from sklearn.model_selection import train_test_split

def conllu_to_flair(path_in, lang, write_corpus: bool = False, write_raw: bool = False):
    data = []
    raw_text = []
    count: int = 0

    with open(path_in) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:                           
            features = []
            raw = []
            
            for tok in line:
                if tok['upos'] != "_":
                    tok['form'] = tok['form'].replace("-", "")
                    combined = tok['form'] + " " + tok['upos'] + " " + str(tok['head']) + " " + tok['deprel']
                    features.append(combined)
                    raw.append(tok['form'])
                    count += 1

            features = "\n".join(features)
            raw = " ".join(raw)

            data.append(features)
            raw_text.append(raw)

    
    if write_corpus == True:
        train, validtext = train_test_split(data, random_state=42, test_size=.2)
        test, dev = train_test_split(validtext, random_state=42, test_size=0.5)

        all_in_one = "\n\n".join(data)
        dev = "\n\n".join(dev)
        test = "\n\n".join(test)
        train = "\n\n".join(train)
        
        data_folder = f'data/{lang}/features'
        
        with open(f'{data_folder}/dev.txt', 'w') as f:
            f.write(dev)
        with open(f'{data_folder}/test.txt', 'w') as f:
            f.write(test)
        with open(f'{data_folder}/train.txt', 'w') as f:
            f.write(train)
        with open(f'{data_folder}/all_in_one.txt', 'w') as f:
            f.write(all_in_one)
        
    if write_raw == True:
        train, validtext = train_test_split(raw_text, random_state=42, test_size=.2)
        test, dev = train_test_split(validtext, random_state=42, test_size=0.5)

        data_emb = f'data/{lang}/embeddings'
        dev_raw = "\n".join(dev)
        test_raw = "\n".join(test)
        train_raw = "\n".join(train)

        with open(f'{data_emb}/valid.txt', 'w') as f:
            f.write(dev_raw)
        with open(f'{data_emb}/test.txt', 'w') as f:
            f.write(test_raw)
        with open(f'{data_emb}/train/train.txt', 'w') as f:
            f.write(train_raw)

    print("Data successfully transformed. Have fun!") 

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

def write_tupi():
    akuntsu = conllu_to_flair('../UD/UD_Akuntsu-TuDeT/aqz_tudet-ud-test.conllu', lang = 'Akuntsu', write_corpus = True, write_raw = True)
    guajajara = conllu_to_flair('../UD/UD_Guajajara-TuDeT/gub_tudet-ud-test.conllu', lang = 'Guajajara', write_corpus = True, write_raw = True)
    kaapor = conllu_to_flair('../UD/UD_Kaapor-TuDeT/urb_tudet-ud-test.conllu', lang = 'Kaapor', write_corpus = True, write_raw = True)
    karo = conllu_to_flair('../UD/UD_Karo-TuDeT/arr_tudet-ud-test.conllu', lang = 'Karo', write_corpus = True, write_raw = True)
    makurap = conllu_to_flair('../UD/UD_Makurap-TuDeT/mpu_tudet-ud-test.conllu', lang = 'Makurap', write_corpus = True, write_raw = True)
    munduruku = conllu_to_flair('../UD/UD_Munduruku-TuDeT/myu_tudet-ud-test.conllu', lang = 'Munduruku', write_corpus = True, write_raw = True)
    tupinamba = conllu_to_flair('../UD/UD_Tupinamba-TuDeT/tpn_tudet-ud-test.conllu', lang = 'Tupinamba', write_corpus = True, write_raw = True)

def make_testset(language):
    data_folder = f'data/{language}/features'
    columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

    corpus: Corpus = ColumnCorpus(data_folder, columns,
                            train_file = 'train.txt',
                            test_file = 'all_in_one.txt',
                            dev_file = 'dev.txt')
    return corpus

def concat(languages: list, folder: str, write_diff_train: bool = True):
    test = []
    dev = []
    train = []
    char_dict = Dictionary()
    char_dict.add_item("<unk>")
    char_dict.add_item("[SEP]")
    char_dict.add_item(" ")
    data_emb = f'data/combi_emb/{folder}'


    for lang in languages:
        lang_text = []
        with open(f'data/{lang}/embeddings/train/train.txt') as file:
            doc = file.read()
            text = doc.split("\n")
            for utt in text:
                lang_text.append(utt)
                for char in utt:
                    if char_dict.get_idx_for_item != 0:
                        char_dict.add_item(char)

        with open(f'data/{lang}/embeddings/valid.txt') as file:
            doc = file.read()
            text = doc.split("\n")
            for utt in text:
                lang_text.append(utt)
                for char in utt:
                    if char_dict.get_idx_for_item != 0:
                        char_dict.add_item(char)

        with open(f'data/{lang}/embeddings/test.txt') as file:
            doc = file.read()
            text = doc.split("\n")
            for utt in text:
                lang_text.append(utt)
                for char in utt:
                    if char_dict.get_idx_for_item != 0:
                        char_dict.add_item(char)

        lang_train, validtext = train_test_split(lang_text, random_state=42, test_size=.2)
        lang_val, lang_test = train_test_split(validtext, random_state=42, test_size=0.5)

        lang_val.append("[SEP]")
        lang_test.append("[SEP]")

        for sentence in lang_val:
            dev.append(sentence)        
        for sentence in lang_test:
            test.append(sentence)

        if write_diff_train == True:
            train_str = "\n".join(lang_train)
            with open(f'{data_emb}/train/{lang}.txt', 'w') as f:
                f.write(train_str)
        else:
            lang_train.append("[SEP]")
            for sentence in lang_train:
                test.append(sentence)

    dev_raw = "\n".join(dev)
    test_raw = "\n".join(test)

    with open(f'{data_emb}/valid.txt', 'w') as f:
        f.write(dev_raw)
    with open(f'{data_emb}/test.txt', 'w') as f:
        f.write(test_raw)

    if write_diff_train == False:
        train_raw = "\n".join(train)
        with open(f'{data_emb}/train/train.txt', 'w') as f:
            f.write(train_raw)

    corpus = TextCorpus(f'data/combi_emb/{folder}', char_dict, forward = True, character_level=True)

    return corpus, char_dict
