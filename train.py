import torch
import numpy as np
from data import DogAndCat, get_train_transforms, get_valid_transforms, get_train_dataloader, get_valid_dataloader
from model import  Network


if __name__ == '__main__':
    epochs = 10
    train_dir = 'dataset/train'
    valid_dir = 'dataset/valid'
    train_dataset = DogAndCat(train_dir, get_train_transforms())
    valid_dataset = DogAndCat(valid_dir, get_valid_transforms())
    train_dataloader = get_train_dataloader(train_dataset, 16)
    valid_dataloader = get_valid_dataloader(valid_dataset, 16)
    cnn_model = Network()
    print(cnn_model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001)
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = cnn_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target).item()
            total += target.size(0)
        train_acc.append(100*correct/total)
        train_loss.append(running_loss/total_step)
        print(f'Train loss: {np.mean(train_loss)}, Train Accuracy: {100*correct/total}\n')
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            cnn_model.eval()
            for data, target in (valid_dataloader):
                outputs = cnn_model(data)
                loss = criterion(outputs, target)
                batch_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                correct_t += torch.sum(pred=target).item()
                total_t += target.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss/len(valid_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'Validation loss: {np.mean(val_loss)}, Validation acc: {100 * correct_t / total_t}\n')
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(cnn_model.state_dict(), 'model_classification_tutorial.pt')
                print('Detected network improvement, saving current model')
        cnn_model.train()

        

