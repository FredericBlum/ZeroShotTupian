from flair.data import MultiCorpus, Sentence
from flair.models import SequenceTagger
from helper_functions import make_testset, write_tupi, make_finetuneset
from flair.trainers import ModelTrainer


################################
### data and dictionaries    ###
################################
write_tupi(write_testset=True)
akuntsu = make_finetuneset(language = 'Akuntsu')
kaapor = make_finetuneset(language = 'Kaapor')
makurap = make_finetuneset(language = 'Makurap')
munduruku = make_finetuneset(language = 'Munduruku')
guajajara = make_finetuneset(language = 'Guajajara')
tupinamba = make_finetuneset(language = 'Tupinamba')
karo = make_finetuneset(language = 'Karo')

print(akuntsu)
print(kaapor)
print(makurap)
print(munduruku)
print(guajajara)
print(tupinamba)
print(karo)
