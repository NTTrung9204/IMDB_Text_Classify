import pandas as pd
import torch
from preprocessing import preprocess_text
from collections import Counter
from prepare_dataset import IMDBDataset
from torch.utils.data import DataLoader, random_split
from MultiHeadAttention import MultiHeadAttention
from train_eval_model import train, evaluate
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


def tokens_to_indices(tokens):
    return [vocab[token] for token in tokens if token in vocab]

if __name__ == "__main__":
    
    D = 128
    max_length = 200
    num_heads = 8
    num_epochs = 10
    learning_rate = 0.001

    data = pd.read_csv("archive/imdb.csv")

    data['tokens'] = data['review'].apply(preprocess_text)

    all_tokens = [token for tokens in data['tokens'] for token in tokens]
    word_counts = Counter(all_tokens)

    max_vocab_size = 20000
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common(max_vocab_size))}

    vocab['<PAD>'] = 0 

    vocab_size = len(vocab)

    print(f"Vocab Size: {vocab_size}")

    data['token_indices'] = data['tokens'].apply(tokens_to_indices)


    data['padded_indices'] = data['token_indices'].apply(
        lambda indices: indices[:max_length] + [0] * (max_length - len(indices)) if len(indices) < max_length else indices[:max_length]
    )

    features = torch.tensor(data['padded_indices'].tolist())
    labels = torch.tensor([1 if label == 'positive' else 0 for label in data['sentiment']])

    dataset = IMDBDataset(features, labels)

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Transformer = MultiHeadAttention(d_embedding=D, s_length=max_length, num_heads=num_heads, vocab_size=vocab_size)
    Transformer.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Transformer.parameters(), lr=learning_rate)

    train_losses, valid_accuracies = train(Transformer, train_loader, valid_loader, criterion, optimizer, device, num_epochs)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.show()

    torch.save(Transformer.state_dict(), "multihead_attention_imdb_model_v2.pth")
    print("Model saved successfully.")