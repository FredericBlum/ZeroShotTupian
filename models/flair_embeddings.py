from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import conllu_to_flair, make_dictionary


################################
### data and dictionaries    ###
################################
dictionary: Dictionary = make_dictionary('data/Shipibo/embeddings/train/train.txt')

# dictionary: Dictionary = Dictionary.load('chars')

# get your corpus, process forward and at the character level
corpus = TextCorpus('data/Shipibo/embeddings',
                    dictionary,
                    character_level=True)

################################
### Language Model           ###
################################
is_forward_lm = True
is_backward_lm = False

language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=512,
                               nlayers=1)

################################
### Tagger and Trainer       ###
################################
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('models/resources/embeddings/sk_word',
                sequence_length=11,
                learning_rate=0.3,
                mini_batch_size=8,
                max_epochs=30)
