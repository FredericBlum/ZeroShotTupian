import torch
import torch.nn.functional as F

from helper_functions import make_onehot_vectors, make_label_vector


class LSTMClassifier(torch.nn.Module):  # inherits from nn.Module!

    def __init__(self,
                 word_dictionary,
                 label_dictionary,
                 embedding_size: int,
                 rnn_hidden_size: int,
                 ):

        super(LSTMClassifier, self).__init__()

        # remember word and label dictionary
        self.word_dictionary = word_dictionary
        self.label_dictionary = label_dictionary

        # remember hyperparameters
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
        # ist one-hot encoding nötig?
        device = 'cuda'
        one_hot_sentence = make_onehot_vectors(sentence, self.word_dictionary)
        one_hot_sentence = one_hot_sentence.to(device)
        embedded = self.word_embeddings(one_hot_sentence)
        #print(embedded)

        lstm_out, (recent_hidden, cell) = self.lstm(embedded)
        #print(recent_hidden)

        tag_space = self.hidden2tag(recent_hidden[0])
        #print(tag_space)
        
        tag_scores = F.log_softmax(tag_space, dim = 1)

        #print(tag_scores)

        # then pass that through log_softmax
        return tag_scores


    def compute_loss(self, log_probabilities_for_each_class, label):
        # make a label vector for the target
        target_vector = make_label_vector(label, self.label_dictionary)
        #print(target_vector)

        # return the negative log likelihood loss
        return F.nll_loss(log_probabilities_for_each_class, target_vector)

    def evaluate(self, test_data) -> float:
        # evaluate the model
        tp: int = 0
        fp: int = 0
        val_loss: int = 0
        val_items = len(test_data)

        with torch.no_grad():

            # go through all test data points
            for instance, label in test_data:

                # send the data point through the model and get a prediction
                log_probs = self.forward(instance)
                loss = self.compute_loss(log_probs, label)
                val_loss += loss

                if torch.argmax(log_probs).item() == self.label_dictionary[label]:
                    tp += 1
                else:
                    fp += 1

            accuracy = tp / (tp + fp)
            av_val_loss = val_loss / val_items
            # print(val_loss, av_val_loss)

            return accuracy, av_val_loss