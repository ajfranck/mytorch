from functions import *

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size)
        # Define the fully connected layer to produce the output
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initialize the hidden state
        hidden = self.init_hidden(x.size(1)).to(device)

        x = x.to(hidden.dtype)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        out = out[:, -1, :]

        return out.float()

    def init_hidden(self, batch_size):
        # Initialize the hidden state with zeros
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden

EPOCHS = 50

# Define hyperparameters
input_size = 27
hidden_size = 64
output_size = 27
learning_rate = 0.01

# Initialize model
model = RNN(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train model
plt.figure(figsize=(8,6))
train_losses = []
test_losses = []
test_accs = []
for t in range(EPOCHS):
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    
    # plot
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    plt.clf()
    plt.plot(np.arange(1, t+2), train_losses, '-', label='train loss')
    plt.plot(np.arange(1, t+2), test_losses, '--', label='test loss')
    # plt.plot(np.arange(1, t+2), test_accs, '-.', label='test acc')
    plt.legend()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.0001)


plt.savefig('RNN.png')

model.eval()  # set model to evaluation mode

loss_function = nn.CrossEntropyLoss()  # or whatever loss function you used during training

total_loss = 0
n_batches = 0

with torch.no_grad():
    for batch in test_dataloader:
        inputs, targets = batch
        predictions = model(inputs.to(device))

        loss = loss_function(predictions, targets.to(device))
        total_loss += loss.item()
        n_batches += 1

average_loss = total_loss / n_batches

# Perplexity is the exponential of the cross-entropy loss
rnn_perplexity = math.exp(average_loss)

print(f"Perplexity RNN: {rnn_perplexity}")


