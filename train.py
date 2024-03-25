import torch
import data
import modeling


def evaluate(encoder_model, decoder_model, eval_data, criterion,
             num_evals=1, print_predictions=True, samples_to_print=1):
    encoder_model.eval()  # Turn on evaluation mode
    decoder_model.eval()  # Turn on evaluation mode

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for batch_idx, batch in enumerate(eval_data):
            if batch_idx >= num_evals:  # Ensure it breaks at num_evals
                break

            T = batch.shape[1]  # Assuming T is the sequence length from inputs

            preds = encoder_model(batch)
            latent = preds[:, -1, :]
            tiled_latent = latent.unsqueeze(1).expand(-1, T, -1)
            tiled_latent = torch.mean(preds, dim=1, keepdims=True)
            tiled_latent = tiled_latent.expand(-1, T, -1)
            reconstructed = decoder_model(tiled_latent)

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
                predicted_labels_reshaped = predicted_labels.view(batch.shape[0], T)
                print("Batch", batch_idx)
                for i in range(min(len(batch), samples_to_print)):
                    print(f"Ground Truth: {''.join(map(str, batch[i].tolist()))}")
                    print(f"Prediction:  {''.join(map(str, predicted_labels_reshaped[i].tolist()))}\n")
                    print(f"Soft prediction:  {''.join(map(str, soft_prediction[i].tolist()))}\n")

    avg_loss = total_loss / min(batch_idx + 1, num_evals)
    accuracy = correct_predictions / total_predictions * 100

    print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy

# When calling evaluate from main, you can now specify to print predictions:
# evaluate(encoder_model, decoder_model, eval_data, criterion, print_predictions=True, samples_to_print=5)


def train(encoder_model, decoder_model, train_data, criterion,
          optimizer, log_interval=1, max_steps=100):
    encoder_model.train()  # turn on train mode
    decoder_model.train()  # turn on train mode
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, batch in enumerate(train_data):
        if batch_idx > max_steps:
            break

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch.to(device)
        # Converting binary tokens into vectors.
        # Input from batch is 0s and 1s of shape [batch_size, T]
        # Output shape should be [batch_size, T, d_model]
        preds = encoder_model(batch) 
        # Take the last prediction as the latent vector.
        # It should have shape [batch_size, d_model]
        latent = preds[:, -1, :]
        # Tile the latent in order to get the desired output
        # size. The output should have shape [batch_size, T, d_model]
        T = preds.shape[1]  # Assuming T is the sequence length from preds
        tiled_latent = latent.unsqueeze(1).expand(-1, T, -1)
        tiled_latent = torch.mean(preds, dim=1, keepdims=True)
        tiled_latent = tiled_latent.expand(-1, T, -1)
        reconstructed = decoder_model(tiled_latent)
        loss = criterion(reconstructed.view(-1, 2), batch.view(-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:  # log_interval could be, e.g., 10
            print(f'Batch: {batch_idx}, Loss: {loss.item()}')


def main():
    # Load dataq
    chunk_size=1024
    split_percentage=0.8
    batch_size=64
    train_data, eval_data = data.create_data_loaders(
        'input.txt',
        chunk_size=chunk_size,
        split_percentage=split_percentage,
        batch_size=batch_size)
    ntokens = 2  # 1 and 0
    # latent_tokens = 512  # 512 latent tokens
    emsize = 128 # embedding dimension
    d_hid = 1024  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 8  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 8  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_model = modeling.TransformerModel(
        ntoken=ntokens,
        d_model=emsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        dropout=dropout,
        include_linear=False,
        max_len=chunk_size).to(device)
    decoder_model = modeling.TransformerModel(
        ntoken=ntokens,
        d_model=emsize,
        d_hid=d_hid,
        nlayers=nlayers,
        nhead=nhead,
        dropout=dropout,
        include_linear=True,
        vector_input=True,
        max_len=chunk_size).to(device)
    # Penalize the model for reconstructing the binary input.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()),
                                 lr=1e-4)
    while 1:
        train(encoder_model, decoder_model, train_data,
            criterion=criterion, optimizer=optimizer)
        evaluate(encoder_model, decoder_model, eval_data,
                criterion=criterion)

if __name__ == "__main__":
    main()  # Call the main function
