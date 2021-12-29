from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

lm_forward = FlairEmbeddings('multi-forward').lm
lm_backward = FlairEmbeddings('multi-backward').lm

dictionary: Dictionary = flair_embedding_forward.dictionary

corpus_for = TextCorpus('data/Shipibo/embeddings',
                    dictionary,
                    is_forward_lm=True,
                    character_level=True)

corpus_back = TextCorpus('data/Shipibo/embeddings',
                    dictionary,
                    is_forward_lm=False,
                    character_level=True)

################################
### Tagger and Trainer       ###
################################
trainer_forward = LanguageModelTrainer(lm_forward, corpus_for)
trainer_backward = LanguageModelTrainer(lm_backward, corpus_back)

trainer_forward.train('models/resources/embeddings/multi_sk_ft/forward',
                sequence_length=100,
                learning_rate=0.3,
                mini_batch_size=8,
                max_epochs=30)

trainer_backward.train('models/resources/embeddings/multi_sk_ft/backward',
                sequence_length=100,
                learning_rate=0.3,
                mini_batch_size=8,
                max_epochs=30)