from flair.data import MultiCorpus, Sentence
from flair.models import SequenceTagger
from helper_functions import make_testset, conllu_to_flair
from flair.trainers import ModelTrainer


################################
### data and dictionaries    ###
################################
akuntsu = make_testset(language = 'Akuntsu')
kaapor = make_testset(language = 'Kaapor')
makurap = make_testset(language = 'Makurap')
munduruku = make_testset(language = 'Munduruku')

corpus = MultiCorpus([akuntsu, 
                        kaapor, 
                        makurap, 
                        munduruku, 
                    ])

################################
### Tagger and Trainer       ###
################################
# tagger = SequenceTagger.load('multi-pos')
tagger = SequenceTagger.load('models/resources/taggers/my-upos-3')
# tagger = SequenceTagger.load('models/resources/taggers/dep_tupi')


trainer = ModelTrainer(tagger, corpus)
#trainer.fine_tune()
trainer.final_test('models/resources/taggers/eval_multi_tupi',
                main_evaluation_metric = ("micro avg", "f1-score"),
                eval_mini_batch_size = 1
                )