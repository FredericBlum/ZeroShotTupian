from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from helper_functions import create_sk_text


create_sk_text('../data/shipibo/orig/train.conllu', '../data/shipibo/flair/train.txt', sep = False)
create_sk_text('../data/shipibo/orig/valid.conllu', '../data/shipibo/flair/valid.txt', sep = False)
create_sk_text('../data/shipibo/orig/test.conllu', '../data/shipibo/flair/test.txt', sep = False)

# init a corpus using column format, data folder and the names of the train, dev and test files
columns = {0: 'text', 1: 'upos'}
data_folder = '../data/shipibo/flair'

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'train.txt',
                              test_file = 'test.txt',
                              dev_file = 'valid.txt')

# print(corpus)

# create label dictionary for a Universal Part-of-Speech tagging task
upos_dictionary = corpus.make_label_dictionary(label_type='upos')
label_type = 'upos'

# print(corpus)
# print(corpus.train[1].to_tagged_string('upos'))


#word_embedding = TransformerWordEmbeddings('xlm-roberta-base') 
word_embedding = TransformerWordEmbeddings('bert-base-multilingual-uncased')

# character embeddings
flair_embedding_forward = FlairEmbeddings('resources/embeddings/sk_forward/best-lm.pt')
# flair_embedding_forward = FlairEmbeddings('multi-forward')
# flair_embedding_backward = FlairEmbeddings('multi-backward')
flair_embedding_backward = FlairEmbeddings('resources/embeddings/sk_backward/best-lm.pt')

embeddings = StackedEmbeddings(embeddings=[word_embedding, flair_embedding_forward, flair_embedding_backward])

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=512,
                        embeddings=embeddings,
                        tag_dictionary=upos_dictionary,
                        tag_type=label_type,
                        use_crf=True)


# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-upos',
                write_weights = True,
               param_selection_mode = True, # necessary because .pt writing is damaged
                learning_rate=0.5,
                mini_batch_size=12,
                max_epochs=40)

# visualize
plotter = Plotter()
#plotter.plot_training_curves('resources/taggers/example-upos/loss.tsv')


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
