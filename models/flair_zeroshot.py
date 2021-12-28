from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import TARSTagger
from flair.trainers import ModelTrainer

from helper_functions import conllu_to_flair

# data and dictionaries
corpus, gold_dict, word_dict = conllu_to_flair('data/shipibo/shipibo-2018jul4.converted.conllu')
label_type = 'upos'
label_dict = corpus.make_label_dictionary(label_type=label_type)

#################
## Embeddings   #
#################

word_embedding = FlairEmbeddings('word_sk')

embeddings = word_embedding
#################
# TARS          #
#################

# switch to a new task (TARS can do multiple tasks so you must define one)
tars = TARSTagger(embeddings=embeddings)

tars.add_and_switch_to_new_task(task_name="pos_tagging",
                                label_dictionary=label_dict,
                                label_type=label_type)

# 7. initialize the text classifier trainer
trainer = ModelTrainer(tars, corpus)

# 8. start the training
trainer.train(base_path='./models/resources/taggers/oneshot_pos',  # path to store the model artifacts
              learning_rate=0.01,  # use very small learning rate
              mini_batch_size=8,
              #mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
              max_epochs=2,
              param_selection_mode = True)

sentence = Sentence('Nato escuelankoxon non onanai , jakon bake inoxon , non nete cuídannoxon')
sentence =tars.predict(sentence)
print(sentence.to_tagged_string("upos"))