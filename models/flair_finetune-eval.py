from flair.data import MultiCorpus, Sentence
from flair.models import SequenceTagger
from helper_functions import make_finetuneset
from flair.trainers import ModelTrainer


################################
### data and dictionaries    ###
################################
akuntsu = make_finetuneset(language = 'Akuntsu')
kaapor = make_finetuneset(language = 'Kaapor')
makurap = make_finetuneset(language = 'Makurap')
munduruku = make_finetuneset(language = 'Munduruku')

karo = make_finetuneset(language = 'Karo')
guajajara = make_finetuneset(language = 'Guajajara')
tupinamba = make_finetuneset(language = 'Tupinamba')

corpus = MultiCorpus([guajajara, karo, tupinamba])
eval_corpus = MultiCorpus([akuntsu, kaapor, makurap, munduruku])

################################
### Tagger and Trainer       ###
################################
tagger = SequenceTagger.load('pos-multi')
#tagger = SequenceTagger.load('models/resources/taggers/my-upos-3/best-model.pt')
# tagger = SequenceTagger.load('models/resources/taggers/dep_tupi/best-model.pt')

trainer = ModelTrainer(tagger, corpus)
#trainer.fine_tune('models/resources/taggers/finetune', mini_batch_size=32, max_epochs=10)

tagger = SequenceTagger.load('models/resources/taggers/finetune/final-model.pt')
eval_set = ModelTrainer(tagger, eval_corpus)
eval_set.final_test('models/resources/taggers/eval_multi_tupi',
                main_evaluation_metric=("micro avg", "f1-score"),
                eval_mini_batch_size=32,
                param_selection_mode=True)
