from flair.data import MultiCorpus, Sentence
from flair.models import SequenceTagger
from helper_functions import make_finetuneset, make_testset
from flair.trainers import ModelTrainer

################################
### data and dictionaries    ###
################################
akuntsu = make_finetuneset(language = 'Akuntsu')
munduruku = make_finetuneset(language = 'Munduruku')

################################
### Evaluation               ###
################################
# tagger = SequenceTagger.load('pos-multi')
tagger = SequenceTagger.load('models/resources/taggers/my-upos-3/final-model.pt')
# tagger = SequenceTagger.load('models/resources/taggers/dep_tupi/best-model.pt')

corpus = munduruku
trainer = ModelTrainer(tagger, corpus)
trainer.fine_tune('models/resources/taggers/finetune', mini_batch_size=32, max_epochs=30, use_final_model_for_eval=False)

akuntsu = make_testset(language = 'Akuntsu')
kaapor = make_testset(language = 'Kaapor')
makurap = make_testset(language = 'Makurap')
munduruku = make_testset(language = 'Munduruku')
eval_corpus = MultiCorpus([akuntsu, kaapor, makurap, munduruku])

tagger = SequenceTagger.load('models/resources/taggers/finetune/best-model.pt')
eval_set = ModelTrainer(tagger, eval_corpus)
eval_set.final_test('models/resources/taggers/eval_multi_tupi',
                main_evaluation_metric=("macro avg", "f1-score"),
                eval_mini_batch_size=32)
