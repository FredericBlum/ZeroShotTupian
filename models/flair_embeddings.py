import torch
import flair
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from helper_functions import conllu_to_flair

# dictionary: Dictionary = make_dictionary(train_text)
# forward or backward LM
is_forward_lm = True
is_backward_lm = False
flair.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the default character dictionary
#dictionary: Dictionary = Dictionary.load('chars')
corpus, gold_dict, word_dict = conllu_to_flair('./data/shipibo/shipibo-2018jul4.converted.conllu')
dictionary = word_dict

# get your corpus, process forward and at the character level
corpus = TextCorpus('./data/shipibo/embeddings',
                    dictionary,
                    character_level=False)

# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=512,
                               nlayers=1)

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('models/resources/embeddings/sk_word',
                sequence_length=9,
	        learning_rate=0.5,
                mini_batch_size=16,
                max_epochs=30)
