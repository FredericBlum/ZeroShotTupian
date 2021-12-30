from flair.data import MultiCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from helper_functions import conllu_to_flair
from flair.visual.training_curves import Plotter


################################
### data and dictionaries    ###
################################
#akuntsu = conllu_to_flair('../UD/UD_Akuntsu-TuDeT/aqz_tudet-ud-test.conllu', lang = 'Akuntsu')
guajajara = conllu_to_flair('../UD/UD_Guajajara-TuDeT/gub_tudet-ud-test.conllu', lang = 'Guajajara')
#kaapor = conllu_to_flair('../UD/UD_Kaapor-TuDeT/urb_tudet-ud-test.conllu', lang = 'Kaapor')
karo = conllu_to_flair('../UD/UD_Karo-TuDeT/arr_tudet-ud-test.conllu', lang = 'Karo')
#makurap = conllu_to_flair('../UD/UD_Makurap-TuDeT/mpu_tudet-ud-test.conllu', lang = 'Makurap')
#munduruku = conllu_to_flair('../UD/UD_Munduruku-TuDeT/myu_tudet-ud-test.conllu', lang = 'Munduruku')
tupinamba = conllu_to_flair('../UD/UD_Tupinamba-TuDeT/tpn_tudet-ud-test.conllu', lang = 'Tupinamba')

corpus = MultiCorpus([#akuntsu, 
                        guajajara, 
                        #kaapor, 
                        karo, 
                        #makurap, 
                        #munduruku, 
                        tupinamba])

label_type = 'upos'
upos_dictionary = corpus.make_label_dictionary(label_type=label_type)

################################
### Embeddings               ###
################################
tf_embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased', fine_tune=True, layers='-1')

# flair_embedding_forward = FlairEmbeddings('multi-forward')
# flair_embedding_backward = FlairEmbeddings('multi-backward')

#embeddings = StackedEmbeddings(embeddings=[# tf_embeddings,
#                                           flair_embedding_forward, flair_embedding_backward])
embeddings = tf_embeddings

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
                learning_rate=0.5,
                mini_batch_size=16,
                max_epochs=100)

###############################
### Visualizations          ###
###############################
plotter = Plotter()
plotter.plot_training_curves('models/resources/taggers/sk_pos/loss.tsv')