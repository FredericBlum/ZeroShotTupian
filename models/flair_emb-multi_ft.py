def finetune_multi(lang):
    lm_forward = FlairEmbeddings('multi-forward').lm
    lm_backward = FlairEmbeddings('multi-backward').lm

    dictionary: Dictionary = lm_forward.dictionary

    corpus_for = TextCorpus(f'data/{lang}/embeddings', dictionary, character_level=True)
    corpus_back = TextCorpus(f'data/{lang}/embeddings', dictionary, False, character_level=True)

    trainer_forward = LanguageModelTrainer(lm_forward, corpus_for)
    trainer_backward = LanguageModelTrainer(lm_backward, corpus_back)

    trainer_forward.train(f'models/resources/embeddings/{lang}/forward',
                    sequence_length=100,
                    learning_rate=0.5,
                    mini_batch_size=1,
                    max_epochs=1)

    trainer_backward.train(f'models/resources/embeddings/{lang}/backward',
                    sequence_length=100,
                    learning_rate=0.5,
                    mini_batch_size=1,
                    max_epochs=1)