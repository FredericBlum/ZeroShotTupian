from flair.data import MultiCorpus, Sentence
from flair.models import SequenceTagger
from helper_functions import make_testset, write_tupi, conllu_to_flair
from flair.trainers import ModelTrainer


################################
### data and dictionaries    ###
################################
write_tupi(write_testset=True)
akuntsu = make_testset(language = 'Akuntsu')
kaapor = make_testset(language = 'Kaapor')
makurap = make_testset(language = 'Makurap')
munduruku = make_testset(language = 'Munduruku')
guajajara = make_testset(language = 'Guajajara')
tupinamba = make_testset(language = 'Tupinamba')
karo = make_testset(language = 'Karo')

print(akuntsu)
print(kaapor)
print(makurap)
print(munduruku)
print(guajajara)
print(tupinamba)
print(karo)
