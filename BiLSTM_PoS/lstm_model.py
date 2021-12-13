import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from helper_functions import make_onehot_vectors, make_label_vector


class LSTMClassifier(torch.nn.Module):

    def __init__(self,
                 word_dictionary,
                 label_dictionary,
                 embedding_size: int,
                 rnn_hidden_size: int,
                 ):

        super(LSTMClassifier, self).__init__()

        # word and label dictionary
        self.word_dictionary = word_dictionary
        self.label_dictionary = label_dictionary

        # hyperparameters
        self.embedding_size = embedding_size
        self.rnn_hidden_size = rnn_hidden_size

        # embeddings
        self.word_embeddings = torch.nn.Embedding(len(self.word_dictionary), self.embedding_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(self.embedding_size, self.rnn_hidden_size, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(self.rnn_hidden_size, len(self.label_dictionary))

    def forward(self, sentence):
        one_hot_sentence = make_onehot_vectors(sentence, self.word_dictionary)
        embedded = self.word_embeddings(one_hot_sentence)
        lstm_out, (recent_hidden, cell) = self.lstm(embedded)
        tag_space = self.hidden2tag(recent_hidden[0])
        tag_scores = F.log_softmax(tag_space, dim = 1)

        return tag_scores


    def compute_loss(self, log_probabilities_for_each_class, label):
        target_vector = make_label_vector(label, self.label_dictionary)

        return F.nll_loss(log_probabilities_for_each_class, target_vector)

    def evaluate(self, test_data) -> float:
        tp: int = 0
        fp: int = 0
        val_loss: int = 0
        val_items = len(test_data)

        test_labels = []
        pred_labels = []

        with torch.no_grad():
            for instance, label in test_data:

                log_probs = self.forward(instance)
                loss = self.compute_loss(log_probs, label)
                val_loss += loss

                if torch.argmax(log_probs).item() == self.label_dictionary[label]:
                    tp += 1
                else:
                    fp += 1

                test_labels.append(self.label_dictionary[label])
                pred_labels.append(torch.argmax(log_probs).item())

            accuracy = tp / (tp + fp)
            av_val_loss = val_loss / val_items

            f1_matrix = classification_report(test_labels, pred_labels, zero_division = "warn")

            return accuracy, av_val_loss, f1_matrix