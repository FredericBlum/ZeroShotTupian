from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from helper_functions import conllu_to_flair


# data and dictionaries
corpus, gold_dict, word_dict = conllu_to_flair('../UD/UD:Karo-TuDeT/arr_tudet-ud-test.conllu')
upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

## Embeddings
# word_embedding = TransformerWordEmbeddings('xlm-roberta-base') 
word_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')

# character embeddings
flair_embedding_forward = FlairEmbeddings('multi-forward')
flair_embedding_backward = FlairEmbeddings('multi-backward')

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
trainer.train('resources/taggers/example-karo',
                write_weights = True,
                param_selection_mode = True, # necessary because .pt writing is damaged
                learning_rate=0.5,
                mini_batch_size=12,
                max_epochs=40)

# visualize
plotter = Plotter()
plotter.plot_training_curves('models/resources/taggers/example-karo/loss.tsv')
