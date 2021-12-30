from flair.data import MultiCorpus, Sentence
from flair.models import SequenceTagger
from helper_functions import conllu_to_flair
from flair.tokenization import SegtokSentenceSplitter
from flair.trainers import ModelTrainer


################################
### data and dictionaries    ###
################################
akuntsu = conllu_to_flair('../UD/UD_Akuntsu-TuDeT/aqz_tudet-ud-test.conllu', lang = 'Akuntsu')
#guajajara = conllu_to_flair('../UD/UD_Guajajara-TuDeT/gub_tudet-ud-test.conllu', lang = 'Guajajara')
#kaapor = conllu_to_flair('../UD/UD_Kaapor-TuDeT/urb_tudet-ud-test.conllu', lang = 'Kaapor')
#karo = conllu_to_flair('../UD/UD_Karo-TuDeT/arr_tudet-ud-test.conllu', lang = 'Karo')
#makurap = conllu_to_flair('../UD/UD_Makurap-TuDeT/mpu_tudet-ud-test.conllu', lang = 'Makurap')
#munduruku = conllu_to_flair('../UD/UD_Munduruku-TuDeT/myu_tudet-ud-test.conllu', lang = 'Munduruku')
#tupinamba = conllu_to_flair('../UD/UD_Tupinamba-TuDeT/tpn_tudet-ud-test.conllu', lang = 'Tupinamba')

corpus = MultiCorpus([akuntsu, 
                        #guajajara, 
                        #kaapor, 
                        #karo, 
                        #makurap, 
                        #munduruku, 
                        #tupinamba
                    ])

################################
### Tagger and Trainer       ###
################################
tagger = SequenceTagger.load('multi-pos')


trainer = ModelTrainer(tagger, corpus)

#trainer.fine_tune()
trainer.final_test('models/resources/taggers/eva-akuntsu',
                main_evaluation_metric = ("micro avg", "f1-score"),
                eval_mini_batch_size = 1
                #train_with_dev=True,
                #monitor_train=True,
                #monitor_test=True,
                #patience=3,
                #anneal_with_restarts=True,
                #learning_rate=0.5,
                #mini_batch_size=16,
                #max_epochs=100
                )