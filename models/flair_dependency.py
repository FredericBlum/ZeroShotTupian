from flair.datasets.conllu import CoNLLUCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings

import flair.models as fl

corpus = CoNLLUCorpus(data_folder = '../data/shipibo/orig',
                    train_file = 'train.conllu',
                    test_file = 'test.conllu',
                    dev_file = 'valid.conllu')


dependency_dictionary = corpus.make_label_dictionary(label_type='head')
label_type = 'dependency'

# word embeddings
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')
# alternatives: xlm-roberta-large, xlm-roberta-base

# character embeddings
flair_embedding_forward = FlairEmbeddings('multi-forward')
flair_embedding_backward = FlairEmbeddings('multi-backward')


embeddings = StackedEmbeddings(embeddings=[bert_embedding, flair_embedding_forward, flair_embedding_backward])

# 5. initialize sequence tagger
tagger = fl.DependencyParser(lstm_hidden_size = 512,
                        embeddings=embeddings,
                        tag_dictionary=dependency_dictionary,
                        tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-dependency',
                #write_weights = True,
                use_final_model_for_eval = True, # necessary because .pt writing is damaged
                learning_rate=0.1,
                mini_batch_size=8,
                max_epochs=20)

# visualize
plotter = Plotter()
plotter.plot_training_curves('loss.txt')
#plotter.plot_weights('weights.txt')