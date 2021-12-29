from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from helper_functions import conllu_to_flair

################################
### data and dictionaries    ###
################################
corpus, gold_dict, word_dict = conllu_to_flair('./data/Shipibo/shipibo-2018jul4.converted.conllu', lang = 'Shipibo')
upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

################################
### Embeddings               ###
################################
# word_embedding = TransformerWordEmbeddings('xlm-roberta-base') 
word_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')
# word_embedding = TransformerWordEmbeddings('sk_word')

flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/sk_forward/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/sk_backward/best-lm.pt')
# flair_embedding_forward = FlairEmbeddings('multi-forward')
# flair_embedding_backward = FlairEmbeddings('multi-backward')

embeddings = StackedEmbeddings(embeddings=[word_embedding, flair_embedding_forward, flair_embedding_backward])

################################
### Tagger and Trainer       ###
################################
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=upos_dictionary,
                        tag_type=label_type,
                        use_crf=True)

trainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/sk_pos',
                param_selection_mode=True,
                train_with_dev=True,
                learning_rate=0.3,
                mini_batch_size=16,
                max_epochs=30)

###############################
### Visualizations          ###
###############################
plotter = Plotter()
#plotter.plot_training_curves('models/resources/taggers/example-upos/loss.tsv')