from conllu import parse
from torch.utils.data import random_split
import torch
from flair.data import Corpus, Dictionary
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.visual.training_curves import Plotter

def conllu_to_flair(path_in, lang, write_corpus: bool = False, write_raw: bool = False):
    data = []
    raw_text = []
    count: int = 0

    with open(path_in) as file:
        doc = file.read()
        doc = parse(doc)

        for line in doc:                           
            utterance = []
            sent = []
            
            for tok in line:
                if tok['upos'] != "_":
                    tok['form'] = tok['form'].replace("-", "")
                    combined = tok['form'] + " " + tok['upos'] + " " + str(tok['head']) + " " + tok['deprel']
                    utterance.append(combined)
                    sent.append(tok['form'])
                    count += 1

            utt_str = "\n".join(utterance)
            sent_str = "\n".join(sent)
            data.append(utt_str)
            raw_text.append(sent_str)

    data_folder = f'./data/{lang}/flair'
    columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

    if write_corpus == True:
        overall = len(data)
        tenth = int(round((overall/10), 0))
        rest = int(overall - tenth - tenth)
        dev, test, train = random_split(data, [tenth, tenth, rest], generator=torch.Generator().manual_seed(1))

        all_in_one = "\n\n".join(data)
        dev = "\n\n".join(dev)
        test = "\n\n".join(test)
        train = "\n\n".join(train)

        with open(f'{data_folder}/dev.txt', 'w') as f:
            f.write(dev)
        with open(f'{data_folder}/test.txt', 'w') as f:
            f.write(test)
        with open(f'{data_folder}/train.txt', 'w') as f:
            f.write(train)
        with open(f'{data_folder}/all_in_one.txt', 'w') as f:
            f.write(all_in_one)
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file = 'all_in_one.txt',
                                #train_file = 'train.txt',
                                test_file = 'test.txt',
                                dev_file = 'dev.txt')

    if write_raw == True:
        dev_raw, test_raw, train_raw = random_split(raw_text, [tenth, tenth, rest], generator=torch.Generator().manual_seed(1))
        data_emb = f'./data/{lang}/embeddings'
        dev_raw = "\n\n".join(dev_raw)
        test_raw = "\n\n".join(test_raw)
        train_raw = "\n\n".join(train_raw)

        with open(f'{data_emb}/valid.txt', 'w') as f:
            f.write(dev_raw)
        with open(f'{data_emb}/test.txt', 'w') as f:
            f.write(test_raw)
        with open(f'{data_emb}/train/train.txt', 'w') as f:
            f.write(train_raw)

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

def finetune_multi(lang):
    lm_forward = FlairEmbeddings('multi-forward').lm
    lm_backward = FlairEmbeddings('multi-backward').lm

    dictionary: Dictionary = lm_forward.dictionary

    corpus_for = TextCorpus(f'data/{lang}/embeddings', dictionary, character_level=True)
    corpus_back = TextCorpus(f'data/{lang}/embeddings', dictionary, False, character_level=True)

    trainer_forward = LanguageModelTrainer(lm_forward, corpus_for)
    trainer_backward = LanguageModelTrainer(lm_backward, corpus_back)

    trainer_forward.train(f'models/resources/embeddings/{lang}/forward',
                    sequence_length=100,
                    learning_rate=0.5,
                    mini_batch_size=1,
                    max_epochs=1)

    trainer_backward.train(f'models/resources/embeddings/{lang}/backward',
                    sequence_length=100,
                    learning_rate=0.5,
                    mini_batch_size=1,
                    max_epochs=1)
    plotter = Plotter()
    plotter.plot_training_curves(f'models/resources/embeddings/{lang}/loss.tsv')

def write_tupi():
    akuntsu = conllu_to_flair('../UD/UD_Akuntsu-TuDeT/aqz_tudet-ud-test.conllu', lang = 'Akuntsu', write_corpus = True, write_raw = True)
    guajajara = conllu_to_flair('../UD/UD_Guajajara-TuDeT/gub_tudet-ud-test.conllu', lang = 'Guajajara', write_corpus = True, write_raw = True)
    kaapor = conllu_to_flair('../UD/UD_Kaapor-TuDeT/urb_tudet-ud-test.conllu', lang = 'Kaapor', write_corpus = True, write_raw = True)
    karo = conllu_to_flair('../UD/UD_Karo-TuDeT/arr_tudet-ud-test.conllu', lang = 'Karo', write_corpus = True, write_raw = True)
    makurap = conllu_to_flair('../UD/UD_Makurap-TuDeT/mpu_tudet-ud-test.conllu', lang = 'Makurap', write_corpus = True, write_raw = True)
    munduruku = conllu_to_flair('../UD/UD_Munduruku-TuDeT/myu_tudet-ud-test.conllu', lang = 'Munduruku', write_corpus = True, write_raw = True)
    tupinamba = conllu_to_flair('../UD/UD_Tupinamba-TuDeT/tpn_tudet-ud-test.conllu', lang = 'Tupinamba', write_corpus = True, write_raw = True)