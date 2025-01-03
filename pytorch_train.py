import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('archive/train.csv')

df["context"] = df["Title"] + " " + df["Description"]

df.drop(columns=['Title', 'Description'], inplace=True)

input_data = df["context"].values
labels = df["Class Index"].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(input_data)

X = X.toarray()

X = torch.tensor(X, dtype=torch.float32)

y = torch.tensor(labels, dtype=torch.long)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
n_hidden = 128
n_letters = 26

rnn = RNN(X.shape[1], n_hidden, 4)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()

n_iters = 1000
print_every = 50
plot_every = 100

current_loss = 0
all_losses = []

for iter in range(1, n_iters + 1):
    category = y[iter - 1]
    line = X[iter - 1]

    output, loss = train(category, line)
    current_loss += loss

    if iter % print_every == 0:
        guess = categoryFromOutput(output)
        correct = '✓' if guess == category.item() else '✗ (%s)' % category.item()
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, current_loss, guess, category.item(), input_data[iter - 1], correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn.state_dict(), 'model.pth')



