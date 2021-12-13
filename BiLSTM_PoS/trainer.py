import torch.optim as optim
import torch.randperm as randperm
from torch.utils.data import DataLoader
import copy


def train(model,
          training_data,
          dev_data,
          test_data,
          learning_rate: float,
          max_epochs: int,
          mini_batch_size: int = 1):

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    batch_size = 16

    training_items = len(training_data)
    epoch_accs = dict()
    best_epoch: int = -1 
    best_accuracy = 0
    best_model = type(model)

    for epoch in range(max_epochs):
        print(f"\n - Epoch {epoch}")

        loss_in_epoch: int = 0

        permutation = randperm(training_data.size()[0])

        data_shuffle = DataLoader(training_data, shuffle = True)
        print(training_data)
        print(permutation)
        print(data_shuffle)

        for instance, label in permutation:
            # Step 4a. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 4b. Run our forward pass.
            log_probabilities_for_each_class = model.forward(instance)
            # print(log_probabilities_for_each_class)

            # Step 4c. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = model.compute_loss(log_probabilities_for_each_class, label)
            loss.backward()
            optimizer.step()

            loss_in_epoch += loss
    
        accuracy, av_val_loss = model.evaluate(dev_data)
        epoch_accs[epoch] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
        
        training_loss = loss_in_epoch / training_items
        print(f"training loss: {training_loss}")
        print(f"validation loss: {av_val_loss}")
        print(f"validation accuracy: {accuracy}")

    best_epoch = max(epoch_accs, key = epoch_accs.get)

    test_accuracy, av_val_loss, f1_matrix = best_model.evaluate(test_data)
    print(f"\n - Training complete.\nBest validation accuracy was observed at epoch {best_epoch}")
    print(f"the final model accuracy is: {test_accuracy}")
    print(f1_matrix)

    return test_accuracy