import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        label = int(parts[0])
        words = parts[1:]
        data.append((words, label))
    return data

def build_vocab(data, min_freq=1):
    word_freq = {}
    for words, _ in data:
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=30):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, label = self.samples[idx]
        ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in words]
        if len(ids) < self.max_len:
            ids += [self.vocab['<PAD>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids), torch.tensor(label)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embed_dim)) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max(p, dim=2)[0] for p in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_classes=2):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])
        return self.fc(x)
    
class TextMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):
        super(TextMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).unsqueeze(2)
        masked_embed = embedded * mask
        sum_embed = masked_embed.sum(dim=1) 
        lengths = mask.sum(dim=1).clamp(min=1)
        avg_embed = sum_embed / lengths

        x = self.fc1(avg_embed)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                preds = torch.argmax(outputs, 1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.tolist())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} - Validation Accuracy: {acc:.4f}")

def evaluate_model(model, test_loader, name="Model"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, 1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"{name} Test Accuracy: {acc:.4f}")
    print(f"{name} Test Precision: {precision:.4f}")
    print(f"{name} Test Recall: {recall:.4f}")
    print(f"{name} Test F1 Score: {f1:.4f}")

def main():
    train_path = 'data/train.txt'
    test_path = 'data/test.txt'
    valid_path = 'data/validation.txt'

    train_data = load_data(train_path)
    val_data = load_data(valid_path)
    test_data = load_data(test_path)

    vocab = build_vocab(train_data)

    train_ds = TextDataset(train_data, vocab)
    val_ds = TextDataset(val_data, vocab)
    test_ds = TextDataset(test_data, vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # CNN
    print("\nTraining CNN model:")
    cnn_model = TextCNN(vocab_size=len(vocab))
    train_model(cnn_model, train_loader, val_loader)
    evaluate_model(cnn_model, test_loader, name="CNN")

    # RNN
    print("\nTraining RNN model:")
    rnn_model = TextRNN(vocab_size=len(vocab))
    train_model(rnn_model, train_loader, val_loader)
    evaluate_model(rnn_model, test_loader, name="RNN")

    # MLP
    print("\nTraining MLP model:")
    mlp_model = TextMLP(vocab_size=len(vocab))
    train_model(mlp_model, train_loader, val_loader)
    evaluate_model(mlp_model, test_loader, name="MLP")

if __name__ == "__main__":
    main()