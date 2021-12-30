from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import conllu_to_flair, concat


################################
### data and dictionaries    ###
################################
dictionary: Dictionary = Dictionary.load('chars')

is_forward_lm = True
is_backward_lm = False

corpus_3 = concat(['Guajajara', 'Tupinamba', 'Karo'], folder = "3")
corpus_7 = concat(['Guajajara', 'Tupinamba', 'Karo', 'Munduruku', 'Kaapor', 'Akuntsu', 'Makurap'], folder = "7")

################################
### Language Model           ###
################################
language_model_for = LanguageModel(dictionary, is_forward_lm, hidden_size=512, nlayers=1)
language_model_back = LanguageModel(dictionary, is_backward_lm, hidden_size=512, nlayers=1)

################################
### Trainers                 ###
################################
trainer = LanguageModelTrainer(language_model_for, corpus_3)
trainer.train('models/resources/embeddings/tupi_3_for', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=100)

trainer = LanguageModelTrainer(language_model_back, corpus_3)
trainer.train('models/resources/embeddings/tupi_3_back', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=100)

trainer = LanguageModelTrainer(language_model_for, corpus_7)
trainer.train('models/resources/embeddings/tupi_7_for', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=100)

trainer = LanguageModelTrainer(language_model_back, corpus_7)
trainer.train('models/resources/embeddings/tupi_7_back', sequence_length=50,learning_rate = 1, mini_batch_size=16, max_epochs=100)
