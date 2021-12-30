from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from helper_functions import conllu_to_flair

################################
### data and dictionaries    ###
################################
corpus = conllu_to_flair('./data/Shipibo/shipibo-2018jul4.converted.conllu', lang = 'Guajajara')
upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

################################
### Embeddings               ###
################################
tf_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased', fine_tune=True, layers='-1')

#flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/sk_forward/best-lm.pt')
#flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/sk_backward/best-lm.pt')
#flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/multi_sk_ft/forward/best-lm.pt')
#flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/multi_sk_ft/backward/best-lm.pt')

#embeddings = StackedEmbeddings(embeddings=[#tf_embedding,
                                           #flair_embedding_forward, flair_embedding_backward])
embeddings = tf_embedding
################################
### Tagger and Trainer       ###
################################
tagger = SequenceTagger(hidden_size=16,
                        embeddings=embeddings,
                        tag_dictionary=upos_dictionary,
                        tag_type=label_type,
                        use_crf=True)

trainer = ModelTrainer(tagger, corpus)

trainer.train('models/resources/taggers/Guajajara',
                train_with_dev=True,
                monitor_train=True,
                monitor_test=True,
                patience=3,
                anneal_with_restarts=True,
                #use_tensorboard=True,
                #tensorboard_log_dir='models/resources/taggers/sk_pos/tensorboard',
                #metrics_for_tensorboard=[("macro avg", "precision"), ("macro avg", "f1-score")],
                learning_rate=1,
                mini_batch_size=1,
                max_epochs=1)

###############################
### Visualizations          ###
###############################
plotter = Plotter()
plotter.plot_training_curves('models/resources/taggers/Guajajara/loss.tsv')