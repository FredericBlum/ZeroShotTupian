from flair.data import MultiCorpus, Sentence
from flair.models import SequenceTagger
from helper_functions import make_testset, write_tupi, make_finetuneset
from flair.trainers import ModelTrainer


################################
### data and dictionaries    ###
################################
write_tupi(write_testset=True, deprel=True, write_corpus=True)
