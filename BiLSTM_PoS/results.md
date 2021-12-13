# Results of different models and hyperparameters

| Approach | Model Parameters | Learning Rate | Max Epochs | Accuracy | Average |
|  ---  | --- | --- | --- | --- |  --- |
| BiLSTM | hidden_size = 512, unk = 0, sep = F  | 0.01  |  20   |  0.88   |    |
| BiLSTM | hidden_size = 512, unk = 0, sep = T  | 0.01  |  20   |  0.64   |    |
| BiLSTM | hidden_size = 512, unk = 0, sep = F  | 0.002  |  30   |  0.885   |    |
| BiLSTM | hidden_size = 1024, unk = 0, sep = F  | 0.001  |  50   |  0.   |  0.857 adam, 0.865 sgd  |

