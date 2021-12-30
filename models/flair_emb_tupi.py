from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import conllu_to_flair


################################
### data and dictionaries    ###
################################
dictionary: Dictionary = Dictionary.load('chars')

is_forward_lm = True
is_backward_lm = False

akuntsu = TextCorpus('data/Akuntsu/embeddings', dictionary, is_forward_lm, character_level=True)
kaapor = TextCorpus('data/Kaapor/embeddings', dictionary, is_forward_lm, character_level=True)
karo = TextCorpus('data/Karo/embeddings', dictionary, is_forward_lm, character_level=True)
guajajara = TextCorpus('data/Guajajara/embeddings', dictionary, is_forward_lm, character_level=True)
makurap = TextCorpus('data/Makurap/embeddings', dictionary, is_forward_lm, character_level=True)
munduruku = TextCorpus('data/Munduruku/embeddings', dictionary, is_forward_lm, character_level=True)
tupinamba = TextCorpus('data/Tupinamba/embeddings', dictionary, is_forward_lm, character_level=True)

corpus_3 = MultiCorpus([guajajara, karo, tupinamba])
corpus_7 = MultiCorpus([akuntsu, guajajara, kaapor, karo, makurap, munduruku, tupinamba])

################################
### Language Model           ###
################################
language_model_for = LanguageModel(dictionary, is_forward_lm, hidden_size=512, nlayers=1)
language_model_back = LanguageModel(dictionary, is_backward_lm, hidden_size=512, nlayers=1)

################################
### Trainers                 ###
################################
trainer = LanguageModelTrainer(language_model_for, corpus_3)
trainer.train('models/resources/embeddings/tupi_3_for', sequence_length=100, mini_batch_size=16, max_epochs=100)

trainer = LanguageModelTrainer(language_model_back, corpus_3)
trainer.train('models/resources/embeddings/tupi_3_back', sequence_length=100, mini_batch_size=16, max_epochs=100)

trainer = LanguageModelTrainer(language_model_for, corpus_7)
trainer.train('models/resources/embeddings/tupi_7_for', sequence_length=100, mini_batch_size=16, max_epochs=100)

trainer = LanguageModelTrainer(language_model_back, corpus_7)
trainer.train('models/resources/embeddings/tupi_7_back', sequence_length=100, mini_batch_size=16, max_epochs=100)