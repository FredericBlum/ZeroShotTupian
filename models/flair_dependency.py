from flair.models import SequenceTagger
from flair.datasets.conllu import CoNLLUCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import DependencyParser
from flair.trainers import ModelTrainer

from helper_functions import conllu_to_flair


# data and dictionaries
corpus, gold_dict, word_dict = conllu_to_flair('./data/shipibo/shipibo-2018jul4.converted.conllu')
label_type = 'deprel'
dependency_dictionary = corpus.make_label_dictionary(label_type=label_type)

# word embeddings
word_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')
# alternatives: xlm-roberta-base

# character embeddings
# flair_embedding_forward = FlairEmbeddings('multi-forward')
# flair_embedding_backward = FlairEmbeddings('multi-backward')
flair_embedding_forward = FlairEmbeddings('models/resources/embeddings/sk_forward/best-lm.pt')
flair_embedding_backward = FlairEmbeddings('models/resources/embeddings/sk_backward/best-lm.pt')

embeddings = StackedEmbeddings(embeddings=[word_embedding, flair_embedding_forward, flair_embedding_backward])

# 5. initialize sequence tagger
tagger = DependencyParser(lstm_hidden_size = 512,
                        token_embeddings=embeddings,
                        relations_dictionary=dependency_dictionary,
                        tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('models/resources/taggers/example-dependency',
                use_final_model_for_eval = True,
                learning_rate=0.1,
                mini_batch_size=8,
                max_epochs=20)

sentence = Sentence('Nato escuelankoxon non onanai , jakon bake inoxon , non nete cuídannoxon')
dep_parser_model: DependencyParser = DependencyParser.load('resources/taggers/example-dependency/best_model.pt')
dep_parser_model.predict(sentence, print_tree=True)

# visualize
plotter = Plotter()
plotter.plot_training_curves('loss.txt')
#plotter.plot_weights('weights.txt')
