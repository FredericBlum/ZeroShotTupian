from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import make_word_dictionary, read_conllu_utt



#dictionary: Dictionary = make_word_dictionary(train_text)
# make_word_dictionary
# are you training a forward or backward LM?
is_forward_lm = True
is_backward_lm = False

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')


# get your corpus, process forward and at the character level
corpus = TextCorpus('../data/shipibo/embeddings',
                    dictionary,
                    #is_forward_lm,
                    character_level=True)


# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=512,
                               nlayers=1)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('resources/embeddings/sk_forward',
              sequence_length=9,
	      learning_rate = 1,
              mini_batch_size=12,
              max_epochs=20)
