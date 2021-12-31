from flair.models import SequenceTagger
from flair.data import Sentence
from flair.datasets.conllu import CoNLLUCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import DependencyParser
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from helper_functions import conllu_to_flair


################################
### data and dictionaries    ###
################################
guajajara = conllu_to_flair('data/UD/UD_Guajajara-TuDeT/gub_tudet-ud-test.conllu', lang = 'Guajajara')
#akuntsu = conllu_to_flair('data/UD/UD_Akuntsu-TuDeT/aqz_tudet-ud-test.conllu', lang = 'Akuntsu')
#kaapor = conllu_to_flair('data/UD/UD_Kaapor-TuDeT/urb_tudet-ud-test.conllu', lang = 'Kaapor')
#makurap = conllu_to_flair('data/UD/UD_Makurap-TuDeT/mpu_tudet-ud-test.conllu', lang = 'Makurap')
#munduruku = conllu_to_flair('data/UD/UD_Munduruku-TuDeT/myu_tudet-ud-test.conllu', lang = 'Munduruku')
karo = conllu_to_flair('data/UD/UD_Karo-TuDeT/arr_tudet-ud-test.conllu', lang = 'Karo')

tupinamba = conllu_to_flair('data/UD/UD_Tupinamba-TuDeT/tpn_tudet-ud-test.conllu', lang = 'Tupinamba')

label_type = 'deprel'
dictionary = corpus.make_label_dictionary(label_type='deprel')

################################
### Embeddings               ###
################################
#transformer_embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased', fine_tune=True, layers='-1')
flair_embedding_forward = FlairEmbeddings('multi-forward')
flair_embedding_backward = FlairEmbeddings('multi-backward')

embeddings = StackedEmbeddings(embeddings=[#transformer_embeddings, 
                                           flair_embedding_forward, flair_embedding_backward])

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
                learning_rate=1,
                mini_batch_size=16,
                max_epochs=150,
                monitor_train=True,
                monitor_test=True,
                anneal_with_restarts=True,
                # use_tensorboard=True,
                # tensorboard_log_dir='models/resources/taggers/sk_pos/tensorboard',
                # metrics_for_tensorboard=[("macro avg", "precision"), ("macro avg", "f1-score")],
                patience=3)

###############################
### Visualizations          ###
###############################
plotter = Plotter()
plotter.plot_training_curves('models/resources/taggers/dep_tupi/loss.tsv')