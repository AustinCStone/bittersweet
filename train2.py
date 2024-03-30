import torch
import data
import modeling
from torchviz import make_dot


class MLPAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(MLPAutoencoder, self).__init__()
        
        self.encoder_layers = torch.nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            self.encoder_layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        
        self.decoder_layers = torch.nn.ModuleList()
        for hidden_dim in reversed(hidden_dims[:-1]):
            self.decoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            self.decoder_layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        self.decoder_layers.append(torch.nn.Linear(prev_dim, input_dim * 2))

    def forward(self, x):
        encoded = x.float()
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        
        reconstructed = encoded
        for layer in self.decoder_layers[:-1]:
            reconstructed = layer(reconstructed)
        
        reconstructed = self.decoder_layers[-1](reconstructed)
        reconstructed = reconstructed.view(x.shape[0], x.shape[1], 2)
        reconstructed = torch.softmax(reconstructed, dim=-1)
        
        return reconstructed

def evaluate(model, eval_data, criterion,
             num_evals=1, print_predictions=True, samples_to_print=1):
    model.eval()  # Turn on evaluation mode

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for batch_idx, batch in enumerate(eval_data):
            if batch_idx >= num_evals:  # Ensure it breaks at num_evals
                break

            reconstructed = model(batch)
            reconstructed_flat = reconstructed.view(-1, 2)
            targets_flat = batch.view(-1).long()
            loss = criterion(reconstructed_flat, targets_flat)
            total_loss += loss.item()
            _, predicted_labels = torch.max(reconstructed_flat, 1)
            soft_prediction = torch.nn.Softmax(dim=-1)(reconstructed)
            correct_predictions += (predicted_labels == targets_flat).sum().item()
            total_predictions += targets_flat.size(0)

            # Optionally print predictions and ground truths
            if print_predictions and batch_idx < samples_to_print:
                predicted_labels_reshaped = predicted_labels.view(batch.shape)
                print("Batch", batch_idx)
                for i in range(min(len(batch), samples_to_print)):
                    print(f"Ground Truth: {''.join(map(str, batch[i].tolist()))}")
                    print(f"Prediction:  {''.join(map(str, predicted_labels_reshaped[i].tolist()))}\n")
                    print(f"Soft prediction:  {''.join(map(str, soft_prediction[i].tolist()))}\n")

    avg_loss = total_loss / min(batch_idx + 1, num_evals)
    accuracy = correct_predictions / total_predictions * 100

    print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy

def train(model, train_data, criterion,
          optimizer, log_interval=1, max_steps=1000):
    model.train()  # turn on train mode
    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(train_data):
        if batch_idx > max_steps:
            break

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch = batch.to(device)

        reconstructed = model(batch)
        loss = criterion(reconstructed.view(-1, 2), batch.view(-1).long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:  # log_interval could be, e.g., 10
            print(f'Batch: {batch_idx}, Loss: {loss.item()}')

def main():
    # Load data
    chunk_size = 128  # Encode just 8 bits sequence length.
    split_percentage = 0.8  # Use 80% of data for training.
    batch_size = 128
    train_data, eval_data = data.create_data_loaders(
        'input.txt',
        chunk_size=chunk_size,
        split_percentage=split_percentage,
        batch_size=batch_size)
    input_dim = chunk_size
    hidden_dims = [128, 256, 512, 1024, 512, 256, 128, 64, 32]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPAutoencoder(input_dim, hidden_dims).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    while True:
        train(model, train_data, criterion=criterion, optimizer=optimizer)
        evaluate(model, eval_data, criterion=criterion)

if __name__ == "__main__":
    main()  # Call the main function