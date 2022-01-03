from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN, UD_FRENCH, UD_ITALIAN, UD_POLISH, UD_DUTCH, UD_CZECH, \
    UD_DANISH, UD_SPANISH, UD_SWEDISH, UD_NORWEGIAN, UD_FINNISH
from flair.embeddings import StackedEmbeddings, FlairEmbeddings

# 1. make a multi corpus consisting of 12 UD treebanks (in_memory=False here because this corpus becomes large)
corpus = MultiCorpus([
    UD_ENGLISH(in_memory=False),
    UD_GERMAN(in_memory=False),
    UD_DUTCH(in_memory=False),
    UD_FRENCH(in_memory=False),
    UD_ITALIAN(in_memory=False),
    UD_SPANISH(in_memory=False),
    UD_POLISH(in_memory=False),
    UD_CZECH(in_memory=False),
    UD_DANISH(in_memory=False),
    UD_SWEDISH(in_memory=False),
    UD_NORWEGIAN(in_memory=False),
    UD_FINNISH(in_memory=False),
])

# 2. what tag do we want to predict?
tag_type = 'upos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# 4. initialize each embedding we use
embedding_types = [

    # contextual string embeddings, forward
    FlairEmbeddings('multi-forward'),

    # contextual string embeddings, backward
    FlairEmbeddings('multi-backward'),
]

# embedding stack consists of Flair and GloVe embeddings
embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type=tag_type,
                        use_crf=False)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(tagger, corpus)

# 7. run training
trainer.train('resources/taggers/upos-multi',
              train_with_dev=True,
              max_epochs=150)