from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from helper_functions import conllu_to_flair


# data and dictionaries
corpus, gold_dict = conllu_to_flair('../data/shipibo/shipibo-2018jul4.converted.conllu')
upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

## Embeddings
# word_embedding = TransformerWordEmbeddings('xlm-roberta-base') 
word_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')

# character embeddings
flair_embedding_forward = FlairEmbeddings('resources/embeddings/sk_forward/best-lm.pt')
# flair_embedding_forward = FlairEmbeddings('multi-forward')
# flair_embedding_backward = FlairEmbeddings('multi-backward')
flair_embedding_backward = FlairEmbeddings('resources/embeddings/sk_backward/best-lm.pt')

embeddings = StackedEmbeddings(embeddings=[word_embedding, flair_embedding_forward, flair_embedding_backward])

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=512,
                        embeddings=embeddings,
                        tag_dictionary=upos_dictionary,
                        tag_type=label_type,
                        use_crf=True)


# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-upos',
                write_weights = True,
                param_selection_mode = True, # necessary because .pt writing is damaged
                learning_rate=0.5,
                mini_batch_size=12,
                max_epochs=40)

# visualize
plotter = Plotter()
#plotter.plot_training_curves('resources/taggers/example-upos/loss.tsv')
