from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.visual.training_curves import Plotter

tf_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased', fine_tune=True, layers='-1')

dictionary: Dictionary = tf_embedding.dictionary

corpus_for = TextCorpus('data/Shipibo/embeddings',
                    dictionary,
                    character_level=True)

################################
### Tagger and Trainer       ###
################################
trainer_forward = LanguageModelTrainer(tf_embedding, corpus_for)

trainer_forward.train('models/resources/embeddings/sk_bert-ft',
                sequence_length=100,
                learning_rate=0.3,
                mini_batch_size=16,
                max_epochs=40)
                
plotter = Plotter()
plotter.plot_training_curves('models/resources/embeddings/sk_bert-ft/loss.tsv')