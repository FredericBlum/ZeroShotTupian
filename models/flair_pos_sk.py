from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from flair.datasets import UD_ENGLISH

from helper_functions import create_sk_text


create_sk_text('../data/shipibo/orig/shipibo_train.conllu', '../data/shipibo/flair/shipibo_train.txt', sep = False)
create_sk_text('../data/shipibo/orig/shipibo_dev.conllu', '../data/shipibo/flair/shipibo_dev.txt', sep = False)
create_sk_text('../data/shipibo/orig/shipibo_test.conllu', '../data/shipibo/flair/shipibo_test.txt', sep = False)

# init a corpus using column format, data folder and the names of the train, dev and test files
columns = {0: 'text', 1: 'upos'}
data_folder = '../data/shipibo/flair'

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'shipibo_train.txt',
                              test_file = 'shipibo_test.txt',
                              dev_file = 'shipibo_dev.txt')

# print(corpus)

# create label dictionary for a Universal Part-of-Speech tagging task
upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

# print(corpus)
# print(corpus.train[1].to_tagged_string('upos'))

# word embeddings
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')
# alternatives: xlm-roberta-large, xlm-roberta-base

# character embeddings
flair_embedding_forward = FlairEmbeddings('multi-forward')
flair_embedding_backward = FlairEmbeddings('multi-backward')


embeddings = StackedEmbeddings(embeddings=[bert_embedding, flair_embedding_forward, flair_embedding_backward])

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=1024,
                        embeddings=embeddings,
                        tag_dictionary=upos_dictionary,
                        tag_type=label_type,
                        use_crf=True)

# tagger = SequenceTagger.load("pos-multi-fast")
# fertiger tagger, nich so gut, vielleicht zero-shot mal anschauen


# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-upos',
                write_weights = True,
               param_selection_mode = True, # necessary because .pt writing is damaged
                learning_rate=0.05,
                mini_batch_size=4,
                max_epochs=25)

# visualize
plotter = Plotter()
plotter.plot_training_curves('loss.tsv')
#plotter.plot_weights('weights.txt')


##########################
# To-Do: fine-tune embeddings

""" # use first and last subtoken for each word
embeddings = TransformerWordEmbeddings('bert-base-uncased', fine_tune=True, layers='-1')
embeddings.embed(sentence)
print(sentence[0].embedding)
# using top most layer for fine-tuning, thats why "-1"
# fine-tune in training tourine!
# fine tune word embeddings

from flair.embeddings import StackedEmbeddings

# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[flair_forward_embedding, flair_backward_embedding, bert_embedding])


# train gensim for word embedding

# convert fasttext to gensim
import gensim

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('/path/to/fasttext/embeddings.txt', binary=False)
word_vectors.save('/path/to/converted')
 """
