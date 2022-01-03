from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import concat

################################
### data and dictionaries    ###
################################
is_forward_lm = True
is_backward_lm = False
lm_forward = FlairEmbeddings('multi-forward').lm
lm_backward = FlairEmbeddings('multi-backward').lm

dictionary: Dictionary = lm_forward.dictionary

tupinamba = TextCorpus("data/Tupinamba/embeddings", char_dict, is_forward_lm, character_level = True)
guajajara = TextCorpus("data/Guajajara/embeddings", char_dict, is_forward_lm, character_level = True)
karo = TextCorpus("data/Karo/embeddings", char_dict, is_forward_lm, character_level = True)

tupinamba_back = TextCorpus("data/Tupinamba/embeddings", char_dict, is_backward_lm, character_level = True)
guajajara_back = TextCorpus("data/Guajajara/embeddings", char_dict, is_backward_lm, character_level = True)
karo_back = TextCorpus("data/Karo/embeddings", char_dict, is_backward_lm, character_level = True)

################################
### Trainers                 ###
################################
trainer_forward = LanguageModelTrainer(lm_forward, tupinamba)
trainer_backward = LanguageModelTrainer(lm_backward, tupinamba_back)

trainer_forward.train(f'models/resources/embeddings/tupi_individual/tupinamba_for_ft',
                sequence_length=80,
                learning_rate=20,
                mini_batch_size=32,
                max_epochs=40)

trainer_backward.train(f'models/resources/embeddings/tupi_individual/tupinamba_back_ft',
                sequence_length=80,
                learning_rate=20,
                mini_batch_size=32,
                max_epochs=40)