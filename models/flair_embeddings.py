from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import make_dictionary

# dictionary: Dictionary = make_dictionary(train_text)
# forward or backward LM
is_forward_lm = True
is_backward_lm = False

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')


# get your corpus, process forward and at the character level
corpus = TextCorpus('./data/shipibo/embeddings',
                    dictionary,
                    character_level=True)


# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_backward_lm,
                               hidden_size=512,
                               nlayers=1)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('models/resources/embeddings/sk_backward',
                sequence_length=9,
	            learning_rate = 1,
                mini_batch_size=12,
                max_epochs=20)
