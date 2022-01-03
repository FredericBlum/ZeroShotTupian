from flair.models import SequenceTagger
from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import DependencyParser
from flair.trainers import ModelTrainer
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
eval_corpus = MultiCorpus([akuntsu, kaapor, makurap, munduruku])

label_type = 'deprel'
dictionary = corpus.make_label_dictionary(label_type=label_type)
eval_dict = eval_corpus.make_label_dictionary(label_type=label_type)

################################
### Embeddings               ###
################################
flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/tupi_3_for_ft/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/tupi_3_back_ft/best-lm.pt')
embeddings = StackedEmbeddings(embeddings=[flair_embedding_forward, flair_embedding_backward])

################################
### Tagger and Trainer       ###
################################
tagger = DependencyParser(lstm_hidden_size=512,
                        token_embeddings=embeddings,
                        relations_dictionary=dictionary,
                        tag_type=label_type)

trainer = ModelTrainer(tagger, corpus)

trainer.train('models/resources/taggers/dep_tupi',
                train_with_test=True,
                learning_rate=1,
                mini_batch_size=1,
                max_epochs=1,
                monitor_train=True,
                monitor_test=True,
                patience=3)

###############################
### Evaluation              ###
###############################
tagger = DependencyParser.load('models/resources/taggers/dep_tupi/best-model.pt')

trainer = ModelTrainer(tagger, eval_corpus)
trainer.final_test('models/resources/taggers/eval_deprel',
                main_evaluation_metric = ("macro avg", "f1-score"),
                eval_mini_batch_size=4,
                gold_label_dictionary_for_eval=eval_dict)
