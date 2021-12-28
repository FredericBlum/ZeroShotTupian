from flair.models import SequenceTagger
from flair.datasets.conllu import CoNLLUCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import DependencyParser
from flair.trainers import ModelTrainer

from helper_functions import conllu_to_flair


# data and dictionaries
corpus, gold_dict = conllu_to_flair('../data/shipibo/shipibo-2018jul4.converted.conllu')
label_type = 'deprel'
dependency_dictionary = corpus.make_label_dictionary(label_type=label_type)

# word embeddings
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')
# alternatives: xlm-roberta-base

# character embeddings
flair_embedding_forward = FlairEmbeddings('multi-forward')
flair_embedding_backward = FlairEmbeddings('multi-backward')

embeddings = StackedEmbeddings(embeddings=[bert_embedding, flair_embedding_forward, flair_embedding_backward])

# 5. initialize sequence tagger
tagger = DependencyParser(lstm_hidden_size = 512,
                        token_embeddings=embeddings,
                        relations_dictionary=dependency_dictionary,
                        tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-dependency',
                #write_weights = True,
                use_final_model_for_eval = True, # necessary because .pt writing is damaged
                learning_rate=0.1,
                mini_batch_size=8,
                max_epochs=20,
                gold_label_dictionary_for_eval = gold_dict)

sentence = Sentence('Nato escuelankoxon non onanai , jakon bake inoxon , non nete cuídannoxon')
dep_parser_model: DependencyParser = DependencyParser.load('resources/taggers/example-dependency/best_model.pt')
dep_parser_model.predict(sentence, print_tree=True)

# visualize
plotter = Plotter()
plotter.plot_training_curves('loss.txt')
#plotter.plot_weights('weights.txt')