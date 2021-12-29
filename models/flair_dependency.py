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
corpus, gold_dict, word_dict = conllu_to_flair('./data/Shipibo/shipibo-2018jul4.converted.conllu', lang = "Shipibo")
label_type = 'deprel'
dictionary = corpus.make_label_dictionary(label_type='deprel')

################################
### Embeddings               ###
################################
#transformer_embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased', fine_tune=True, layers='-1')
# flair_embedding_forward = FlairEmbeddings('multi-forward')
# flair_embedding_backward = FlairEmbeddings('multi-backward')
flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/sk_forward/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/sk_backward/best-lm.pt')

embeddings = StackedEmbeddings(embeddings=[#transformer_embeddings, 
                                           flair_embedding_forward, flair_embedding_backward])

################################
### Tagger and Trainer       ###
################################
tagger = DependencyParser(lstm_hidden_size=256,
                        token_embeddings=embeddings,
                        relations_dictionary=dictionary,
                        tag_type=label_type)

trainer = ModelTrainer(tagger, corpus)

trainer.train('models/resources/taggers/sk_dep',
                train_with_dev=True,
                learning_rate=0.2,
                mini_batch_size=8,
                max_epochs=70)

###############################
### Visualizations          ###
###############################
plotter = Plotter()
plotter.plot_training_curves('models/resources/taggers/sk_dep/loss.tsv')

# sentence = Sentence('Nato escuelankoxon non onanai , jakon bake inoxon , non nete cuídannoxon')
# dep_parser_model: DependencyParser = DependencyParser.load('models/resources/taggers/sk_dep/best-model.pt')
# dep_parser_model.predict(sentence, print_tree=True)
