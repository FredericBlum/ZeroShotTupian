from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import conllu_to_flair, make_dictionary


################################
### data and dictionaries    ###
################################
dictionary: Dictionary = Dictionary.load('chars-xl')

is_forward_lm = True
is_backward_lm = False

corpus = TextCorpus('data/Guajajara/embeddings_new',
                    dictionary,
                    is_forward_lm,
                    character_level=True)

################################
### Language Model           ###
################################
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=256,
                               nlayers=1)

################################
### Tagger and Trainer       ###
################################
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('models/resources/embeddings/sk_backward',
                sequence_length=30,
                mini_batch_size=1,
                max_epochs=30)