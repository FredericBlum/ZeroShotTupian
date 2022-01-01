from helper_functions import write_tupi, make_word_dictionary
import transformers
from flair.data import Sentence
from flair.datasets import ColumnCorpus

#write_tupi(write_corpus=True)
sentence = Sentence('France is the current world cup winner.')
print(sentence.to_plain_string())



word_dict = make_word_dictionary(["Guajajara", "Tupinamba"])
print(word_dict)