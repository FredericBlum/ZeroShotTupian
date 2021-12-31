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
tupinamba = TextCorpus("data/Tupinamba/embeddings", char_dict, is_forward_lm, character_level = True)
guajajara = TextCorpus("data/Guajajara/embeddings", char_dict, is_forward_lm, character_level = True)
karo = TextCorpus("data/Karo/embeddings", char_dict, is_forward_lm, character_level = True)
corpus_3 = TextCorpus("data/combi_emb/3", char_dict, is_forward_lm, character_level = True)
corpus_7 = TextCorpus("data/combi_emb/7", char_dict, is_forward_lm, character_level = True)

language_model_for = LanguageModel(char_dict, is_forward_lm, hidden_size=256, nlayers=1)
language_model_back = LanguageModel(char_dict, is_backward_lm, hidden_size=256, nlayers=1)

################################
### Trainers                 ###
################################

trainer = LanguageModelTrainer(language_model_for, corpus_3)
trainer.train('models/resources/embeddings/tupi_3_for', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_for, corpus_7)
trainer.train('models/resources/embeddings/tupi_7_for', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_for, guajajara)
trainer.train('models/resources/embeddings/tupi_individual/tupi_guajajara_for', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_for, tupinamba)
trainer.train('models/resources/embeddings/tupi_individual/tupi_tupinamba_for', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_for, karo)
trainer.train('models/resources/embeddings/tupi_individual/tupi_karo_for', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

tupinamba = TextCorpus("data/Tupinamba/embeddings", char_dict, is_backward_lm, character_level = True)
karo = TextCorpus("data/Karo/embeddings", char_dict, is_backward_lm, character_level = True)
guajajara = TextCorpus("data/Guajajara/embeddings", char_dict, is_backward_lm, character_level = True)
corpus_3 = TextCorpus("data/combi_emb/3", char_dict, is_backward_lm, character_level = True)
corpus_7 = TextCorpus("data/combi_emb/7", char_dict, is_backward_lm, character_level = True)

trainer = LanguageModelTrainer(language_model_back, guajajara)
trainer.train('models/resources/embeddings/tupi_individual/tupi_guajajara_back', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_back, tupinamba)
trainer.train('models/resources/embeddings/tupi_individual/tupi_tupinamba_back', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_back, karo)
trainer.train('models/resources/embeddings/tupi_individual/tupi_karo_back', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_back, corpus_3)
trainer.train('models/resources/embeddings/tupi_3_back', sequence_length=80, mini_batch_size=16, learning_rate=20, max_epochs=200)

trainer = LanguageModelTrainer(language_model_back, corpus_7)
trainer.train('models/resources/embeddings/tupi_7_back', sequence_length=80,learning_rate=20, mini_batch_size=16, max_epochs=200)
