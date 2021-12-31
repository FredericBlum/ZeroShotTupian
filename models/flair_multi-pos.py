from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from helper_functions import conllu_to_flair
from flair.visual.training_curves import Plotter



################################
### data and dictionaries    ###
################################
columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

akuntsu = ColumnCorpus('data/Akuntsu/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
guajajara = ColumnCorpus('data/Guajajara/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
kaapor = ColumnCorpus('data/Kaapor/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
karo = ColumnCorpus('data/Karo/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
makurap = ColumnCorpus('data/Makurap/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
munduruku = ColumnCorpus('data/Munduruku/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
tupinamba = ColumnCorpus('data/Tupinamba/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')

corpus = MultiCorpus([#akuntsu, 
                        guajajara, 
                        #kaapor, 
                        karo, 
                        #makurap, 
                        #munduruku, 
                        tupinamba])
print(corpus)

label_type = 'upos'
upos_dictionary = corpus.make_label_dictionary(label_type=label_type)

################################
### Embeddings               ###
################################

flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/tupi_3_for/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/tupi_3_back/best-lm.pt')
#flair_embedding_forward = FlairEmbeddings("multi-forward")
#flair_embedding_backward = FlairEmbeddings("multi-backward")
embeddings = StackedEmbeddings(embeddings=[#tf_embeddings,
                                           flair_embedding_forward, flair_embedding_backward])
# embeddings = TransformerWordEmbeddings("bert-base-multilingual-cased", fine_tune=True, layers="-1")

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
                anneal_with_restarts=True,
                learning_rate=1,
                mini_batch_size=16,
                max_epochs=300)

###############################
### Visualizations          ###
###############################
plotter = Plotter()
plotter.plot_training_curves('models/resources/taggers/sk_pos/loss.tsv')
