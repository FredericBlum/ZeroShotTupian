from flair.models import SequenceTagger
from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import DependencyParser
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter


################################
### data and dictionaries    ###
################################
columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

guajajara = ColumnCorpus('data/Guajajara/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
karo = ColumnCorpus('data/Karo/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
tupinamba = ColumnCorpus('data/Tupinamba/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')

corpus = MultiCorpus([guajajara, 
                    karo, 
                    tupinamba])

label_type = 'deprel'
dictionary = corpus.make_label_dictionary(label_type='deprel')

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
                train_with_dev=True,
                learning_rate=4,
                mini_batch_size=16,
                max_epochs=500,
                monitor_train=True,
                monitor_test=True,
                anneal_with_restarts=True,
                patience=3)

###############################
### Visualizations          ###
###############################
plotter = Plotter()
plotter.plot_training_curves('models/resources/taggers/dep_tupi/loss.tsv')
