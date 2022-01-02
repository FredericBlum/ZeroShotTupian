from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

################################
### data and dictionaries    ###
################################
guajajara = ColumnCorpus('data/Guajajara/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
karo = ColumnCorpus('data/Karo/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')
tupinamba = ColumnCorpus('data/Tupinamba/features', columns, train_file = 'train.txt', test_file = 'test.txt', dev_file = 'dev.txt')

akuntsu = make_testset(language = 'Akuntsu')
kaapor = make_testset(language = 'Kaapor')
makurap = make_testset(language = 'Makurap')
munduruku = make_testset(language = 'Munduruku')

eval_corpus = MultiCorpus([akuntsu, kaapor, makurap, munduruku])
corpus = guajajara

upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

################################
### Embeddings               ###
################################

flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/guajajara_for_ft/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/tupi_individual/guajajara_back_ft/best-lm.pt')

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

trainer.train('models/resources/taggers/tupi_ind',
                train_with_dev=True,
                monitor_train=True,
                monitor_test=True,
                patience=3,
                anneal_with_restarts=True,
                learning_rate=1,
                mini_batch_size=16,
                max_epochs=300)

###############################
### Evaluation              ###
###############################
tagger = SequenceTagger.load('models/resources/taggers/tupi_ind/best-model.pt')

trainer = ModelTrainer(tagger, eval_corpus)
trainer.final_test('models/resources/taggers/eval_multi_tupi',
                main_evaluation_metric = ("micro avg", "f1-score"),
                eval_mini_batch_size = 32)