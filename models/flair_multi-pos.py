from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from helper_functions import make_testset


################################
### data and dictionaries    ###
################################
columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

guajajara = ColumnCorpus('data/Guajajara/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
karo = ColumnCorpus('data/Karo/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
tupinamba = ColumnCorpus('data/Tupinamba/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')

akuntsu = make_testset(language = 'Akuntsu')
kaapor = make_testset(language = 'Kaapor')
makurap = make_testset(language = 'Makurap')
munduruku = make_testset(language = 'Munduruku')

corpus = MultiCorpus([guajajara, karo, tupinamba])
# corpus = karo
eval_corpus = MultiCorpus([akuntsu, kaapor, makurap, munduruku])

label_type = 'upos'
upos_dictionary = corpus.make_label_dictionary(label_type=label_type)

################################
### Embeddings               ###
################################
flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/tupi_3_for_ft/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/tupi_3_back_ft/best-lm.pt')
embeddings = StackedEmbeddings(embeddings=[flair_embedding_forward, flair_embedding_backward])

################################
### Tagger and Trainer       ###
################################
tagger = SequenceTagger(hidden_size=512,
                        embeddings=embeddings,
                        tag_dictionary=upos_dictionary,
                        tag_type=label_type,
                        use_crf=True)

trainer = ModelTrainer(tagger, corpus)

trainer.train('models/resources/taggers/my-upos-3',
                train_with_dev=True,
                monitor_train=True,
                monitor_test=True,
                patience=3,
                learning_rate=1,
                mini_batch_size=32,
                max_epochs=300)

###############################
### Evaluation              ###
###############################
#tagger = SequenceTagger.load('multi-pos')
tagger = SequenceTagger.load('models/resources/taggers/my-upos-3/final-model.pt')

trainer = ModelTrainer(tagger, eval_corpus)
#trainer.fine_tune()
trainer.final_test('models/resources/taggers/eval_multi_tupi',
                main_evaluation_metric = ("macro avg", "f1-score"),
                eval_mini_batch_size = 32)