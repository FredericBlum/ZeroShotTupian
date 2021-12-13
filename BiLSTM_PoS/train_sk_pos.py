import torch

from helper_functions import make_word_dictionary, make_label_dictionary, read_conllu
from lstm_model import LSTMClassifier

torch.manual_seed(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device: {}'.format(device))


# All hyperparameters
hidden_size = 1024
learning_rate = 0.002
max_epochs = 20
unk_threshold = 0

# prepare data
# train_data = read_conllu('../data/shipibo.conllu')
train_data = read_conllu('../data/shipibo.conllu', sep = True)

dev_data = read_conllu('../data/shipibo_valid.conllu', char = True)
test_data = read_conllu('../data/shipibo_test.conllu', char = True)
label_data = read_conllu('../data/shipibo_all.conllu', char = True)


word_dictionary = make_word_dictionary(train_data, unk_threshold = unk_threshold)
label_dictionary, label_freq = make_label_dictionary(label_data)
# print(label_freq)

model = LSTMClassifier(word_dictionary=word_dictionary,
                       label_dictionary=label_dictionary,
                       embedding_size=hidden_size,
                       rnn_hidden_size=hidden_size,
                       )

model.to(device)

import trainer

trainer.train(model=model,
              training_data=train_data,
              dev_data=dev_data,
              test_data=test_data,
              learning_rate=learning_rate,
              max_epochs=max_epochs,
              )