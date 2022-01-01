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

corpus_for = TextCorpus("data/combi_emb/3", dictionary, is_forward_lm, character_level = True)
corpus_back = TextCorpus("data/combi_emb/3", dictionary, is_backward_lm, character_level = True)

################################
### Trainers                 ###
################################
trainer_forward = LanguageModelTrainer(lm_forward, corpus_for)
trainer_backward = LanguageModelTrainer(lm_backward, corpus_back)

trainer_forward.train(f'models/resources/embeddings/tupi_3_for_ft',
                sequence_length=80,
                learning_rate=20,
                mini_batch_size=32,
                max_epochs=20)

trainer_backward.train(f'models/resources/embeddings/tupi_3_back_ft',
                sequence_length=80,
                learning_rate=20,
                mini_batch_size=32,
                max_epochs=20)
