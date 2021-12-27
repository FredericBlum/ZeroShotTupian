from helper_functions import read_conllu_utt


corpus = CoNLLUCorpus(data_folder = '../data/shipibo/orig',
                    train_file = 'train.conllu',
                    test_file = 'test.conllu',
                    dev_file = 'valid.conllu')
print(corpus)

train_text = read_conllu_utt('../data/shipibo/orig/train.conllu')
with open('../data/shipibo/embeddings/train.txt', 'w') as f:
    f.write(train_text)

test_text = read_conllu_utt('../data/shipibo/orig/test.conllu')
with open('../data/shipibo/embeddings/test.txt', 'w') as f:
    f.write(test_text)

dev_text = read_conllu_utt('../data/shipibo/orig/valid.conllu')
with open('../data/shipibo/embeddings/valid.txt', 'w') as f:
    f.write(dev_text)