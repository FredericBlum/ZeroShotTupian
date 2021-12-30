from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import conllu_to_flair, concat


################################
### data and dictionaries    ###
################################
is_forward_lm = True
is_backward_lm = False

corpus_3, char_dict = concat(['Guajajara', 'Tupinamba', 'Karo'], folder = "3")
corpus_7, char_dict = concat(['Guajajara', 'Tupinamba', 'Karo', 'Munduruku', 'Kaapor', 'Akuntsu', 'Makurap'], folder = "7")

################################
### Language Model           ###
################################
tupinamba = TextCorpus("data/Tupinamba/embeddings_new", char_dict, is_forward_lm, character_level = True)
karo = TextCorpus("data/Karo/embeddings_new", char_dict, is_forward_lm, character_level = True)
guajajara = TextCorpus("data/Guajajara/embeddings_new", char_dict, is_forward_lm, character_level = True)

language_model_for = LanguageModel(char_dict, is_forward_lm, hidden_size=128, nlayers=1)
language_model_back = LanguageModel(char_dict, is_backward_lm, hidden_size=128, nlayers=1)

################################
### Trainers                 ###
################################
trainer = LanguageModelTrainer(language_model_for, tupinamba)
trainer.train('models/resources/embeddings_new/tupi_tupinamba', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=200)

""" trainer = LanguageModelTrainer(language_model_for, karo)
trainer.train('models/resources/embeddings_new/tupi_karo', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=200)

trainer = LanguageModelTrainer(language_model_for, guajajara)
trainer.train('models/resources/embeddings_new/tupi_guajajara', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=200)

trainer = LanguageModelTrainer(language_model_for, corpus_3)
trainer.train('models/resources/embeddings_new/tupi_tupinamba', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=200)

trainer = LanguageModelTrainer(language_model_back, corpus_3)
trainer.train('models/resources/embeddings_new/tupi_3_back', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=200)

trainer = LanguageModelTrainer(language_model_for, corpus_7)
trainer.train('models/resources/embeddings_new/tupi_7_for', sequence_length=50, mini_batch_size=16, learning_rate = 1, max_epochs=200)

trainer = LanguageModelTrainer(language_model_back, corpus_7)
trainer.train('models/resources/embeddings_new/tupi_7_back', sequence_length=50,learning_rate = 1, mini_batch_size=16, max_epochs=200) """