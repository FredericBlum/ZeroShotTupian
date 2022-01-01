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

################################
### Tagger and Trainer       ###
################################
tagger = SequenceTagger.load('multi-pos')
#tagger = SequenceTagger.load('models/resources/taggers/my-upos-3/best-model.pt')
# tagger = SequenceTagger.load('models/resources/taggers/dep_tupi/best-model.pt')


trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune('models/resources/taggers/finetune',
                learning_rate=1,
                mini_batch_size=4,
                max_epochs=30)

trainer.final_test('models/resources/taggers/finetune',
                main_evaluation_metric = ("micro avg", "f1-score"),
                eval_mini_batch_size = 32
                )