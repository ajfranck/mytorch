from dataloader import *

def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    for i, (X, y) in enumerate(dataloader):
        # y = y.type(torch.float)
        y = y.type(torch.int64)
        y = y.to(device)
        X = X.to(device)
        # Compute prediction and loss
        # X = X.reshape([1,32])

        pred = model(X)
        pred = pred.squeeze(1)
        # print('pred', pred.shape)
        # print('y', y.shape)
        loss = loss_fn(pred.float(), y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return(loss_fn(pred.float(), y.float()).item())
    

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.type(torch.int64)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.float(), y.float()).item()
            correct += sum(pred.argmax(-1) == y[:,-1])

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

