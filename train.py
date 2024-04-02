# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers==0.0.23post1
# wandb key: 4d89c43f67fc55f37cc6e65e9304ef29b1a454f3
import torch
import numpy as np
import data
import torch.nn.functional as F
import modeling
# from torchviz import make_dot
import wandb

DEBUG=True
USE_WANDB=False


import torch
import torch.nn.functional as F


def diversity_loss(vectors, subsample_size=1000):
    """
    Compute the diversity loss for a batch of vectors with random subsampling to avoid large similarity matrix computations.
    
    Args:
    - vectors (Tensor): A 3D tensor of shape (batch_size, sequence_dim, vector_dim) where each row is a vector.
    - subsample_size (int): The number of vectors to randomly subsample for the diversity calculation.
    
    Returns:
    - loss (Tensor): A scalar tensor representing the diversity loss.
    """
    batch_size, seq_dim, vector_dim = vectors.shape

    # Reshape to treat each vector in the sequence separately
    vectors = vectors.reshape(batch_size * seq_dim, vector_dim)
    
    # Randomly subsample vectors to reduce size
    total_vectors = vectors.shape[0]
    subsample_indices = torch.randperm(total_vectors)[:subsample_size]
    vectors_subsampled = vectors[subsample_indices]

    # Normalize the subsampled vectors to unit length
    vectors_norm = F.normalize(vectors_subsampled, p=2, dim=1)
    
    # Compute the cosine similarity matrix for the subsampled set
    similarity_matrix = torch.matmul(vectors_norm, vectors_norm.T)
    
    # Zero out the diagonal (self-similarity) by subtracting it out
    eye = torch.eye(vectors_subsampled.shape[0], device=vectors.device)
    similarity_matrix = similarity_matrix - eye
    
    # Since we want to minimize similarity, we take the sum of all positive similarities
    positive_similarities = torch.relu(similarity_matrix)
    loss = positive_similarities.sum() / (vectors_subsampled.shape[0] * (vectors_subsampled.shape[0] - 1))

    return loss


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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = batch.to(device)
            T = batch.shape[1]  # Assuming T is the sequence length from inputs

            hard_preds_st, _, _, _ = encoder_model(batch) 
            # latent = preds[:, -1, :]
            # Tile the mean prediction.
            # tiled_latent = latent 
            # tiled_latent = torch.mean(preds, dim=1, keepdims=True)
            # tiled_latent = latent.unsqueeze(1)  

            # tiled_latent = tiled_latent.expand(-1, T, -1)
            reconstructed = decoder_model(hard_preds_st)
            num_classes = reconstructed.shape[-1]
            reconstructed_flat = reconstructed.view(-1, num_classes)
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
                    print(f"Ground Truth: {','.join(map(str, batch[i].tolist()))}")
                    print(f"Prediction:  {','.join(map(str, predicted_labels_reshaped[i].tolist()))}\n")
                    # print(f"Soft prediction:  {''.join(map(str, soft_prediction[i].tolist()))}\n")

    avg_loss = total_loss / min(batch_idx + 1, num_evals)
    accuracy = correct_predictions / total_predictions * 100
    if USE_WANDB:
        wandb.log({'eval_loss': avg_loss, 'eval_accuracy': accuracy})
    print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy

# When calling evaluate from main, you can now specify to print predictions:
# evaluate(encoder_model, decoder_model, eval_data, criterion, print_predictions=True, samples_to_print=5)


def train(encoder_model, decoder_model, train_data, criterion,
          optimizer, log_interval=1, max_steps=100, start_step=0,
          diversity_weight=1.0):
    encoder_model.train()  # turn on train mode
    decoder_model.train()  # turn on train mode
    criterion = torch.nn.CrossEntropyLoss()
    losses = {
        'loss_recon': [],
        'vq_loss': [],
        'commit_loss': [],
        'diversity_loss': [],
    }
    for batch_idx, batch in enumerate(train_data):
        if batch_idx > max_steps:
            break

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch = batch.to(device)
        # Converting binary tokens into vectors.
        # Input from batch is 0s and 1s of shape [batch_size, T]
        # Output shape should be [batch_size, T, d_model]
        optimizer.zero_grad()
        hard_preds_st, hard_preds, soft_preds, tokens = encoder_model(batch) 
        # Take the last prediction as the latent vector.
        # It should have shape [batch_size, d_model]
        #latent = tiled_latent = preds
        # latent = hard_preds[:, -1, :]
        # 0/0
        # Tile the latent in order to get the desired output
        # size. The output should have shape [batch_size, T, d_model]
        # T = hard_preds.shape[1]  # Assuming T is the sequence length from preds
        # Tile the mean prediction.
        # tiled_latent = latent.unsqueeze(1)  
        #tiled_latent = torch.mean(preds, dim=1, keepdims=True)
        # print("Tiled latent size", tiled_latent.shape)
        # print("Tiled latent", tiled_latent)
        # tiled_latent = tiled_latent.expand(-1, T, -1)
        # print("Tiled latent 2", tiled_latent)
        reconstructed = decoder_model(hard_preds_st)
        # yhat = reconstructed
        #make_dot(yhat, params=dict(list(encoder_model.named_parameters()) + list(decoder_model.named_parameters()))).render("rnn_torchviz", format="png")
        num_classes = reconstructed.shape[-1]
        loss_recon = criterion(reconstructed.view(-1, num_classes), batch.view(-1).long())
        loss_div = diversity_loss(soft_preds) * diversity_weight
        loss_vq = F.mse_loss(hard_preds, soft_preds.detach())
        loss_commit = F.mse_loss(soft_preds, hard_preds.detach())
        # TODO add loss weights
        loss = loss_recon + loss_vq + loss_commit + loss_div
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder_model.parameters()) + list(decoder_model.parameters()), 0.5)
        optimizer.step()
        if batch_idx % log_interval == 0:  # log_interval could be, e.g., 10
            print(f'Batch: {batch_idx + start_step}, Loss: {loss.item()}, '
                  f'Recon loss: {loss_recon.item()}, VQ loss: {loss_vq.item()}, Commit loss: {loss_commit.item()}, '
                  f'Diversity loss: {loss_div.item()}')
            if batch_idx % 100 == 0:
                reconstructed_flat = reconstructed.view(-1, num_classes)
                _, predicted_labels = torch.max(reconstructed_flat, 1)
                predicted_labels_reshaped = predicted_labels.view(batch.shape[0], batch.shape[1])
                print("Ground truth:", batch[0])
                print("Latent prediction:", tokens[0])
                print("Reconstructed prediction:", predicted_labels_reshaped[0])
            losses['loss_recon'].append(loss_recon.item())
            losses['vq_loss'].append(loss_vq.item())
            losses['commit_loss'].append(loss_commit.item())
            losses['diversity_loss'].append(loss_div.item())
            if USE_WANDB:
                wandb.log({'train_loss_recon': loss_recon.item(),
                           'train_vq_loss': loss_vq.item(),
                           'train_commit_loss': loss_commit.item(),
                           'train_diversity_loss': loss_div.item()})
    return {k: np.mean(v) for k, v in losses.items()}

def main():
    if DEBUG:
        config = {
            # Load data
            'chunk_size':8, # Encode 8 bytes sequence length.
            'split_percentage':0.8, # Use 80% of data for training.
            'batch_size':8,
            # model hypers
            'lr':1e-4,
            'diversity_weight': 500.0,
            'ntokens':256,  # All bytes.
            'd_model':256,
            'd_hid':512,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
            'nlayers':8,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            'nhead': 2,  # number of heads in ``nn.MultiheadAttention``
            'dropout': 0.2,  # dropout probability
            'num_latent_vectors': 24_000,
            'use_bits': False,
            'compression_factor': 2,
        }
    else:
        config = {
            # Load data
            'chunk_size': 128, # Encode 8 bytes sequence length.
            'split_percentage': 0.8, # Use 80% of data for training.
            'batch_size': 128,
            'lr': 1e-4,
            'diversity_weight': 500.0,
            # model hypers
            'ntokens': 256,  # All bytes.
            'd_model': 512,
            'd_hid': 512,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
            'nlayers':4,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            'nhead': 4,  # number of heads in ``nn.MultiheadAttention``
            'dropout': 0.2,  # dropout probability
            'num_latent_vectors': 124_000,
            'use_bits': False,
            'compression_factor': 2,
        }
    # start a new wandb run to track this script
    if USE_WANDB:
        wandb.init(
            # set the wandb project where this run will be logged
            project="rackitten-tokenizer",
            # track hyperparameters and run metadata
            config=config
        )
    train_data, eval_data = data.create_data_loaders(
        'input.txt',
        chunk_size=config['chunk_size'],
        split_percentage=config['split_percentage'],
        batch_size=config['batch_size'],
        use_bits=config['use_bits'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_model = modeling.TransformerModel(
        ntoken=config['ntokens'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        d_hid=config['d_hid'],
        nlayers=config['nlayers'],
        dropout=config['dropout'],
        include_linear=False,
        use_vq=True,
        num_latent_vectors=config['num_latent_vectors'],
        max_len=config['chunk_size'],
        compression_factor=config['compression_factor']).to(device)
    assert config['d_model'] % config['compression_factor'] == 0
    decoder_model = modeling.TransformerModel(
        ntoken=config['ntokens'],
        d_model=config['d_model'] // config['compression_factor'],
        d_hid=config['d_hid'],
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        dropout=config['dropout'],
        include_linear=True,
        vector_input=True,
        use_vq=False,
        max_len=config['chunk_size']).to(device)
    # Penalize the model for reconstructing the binary input.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()),
                                 lr=config['lr'])
    steps = 0
    for _ in range(1000):
        train_losses = train(encoder_model, decoder_model, train_data,
                             criterion=criterion, optimizer=optimizer,
                             start_step=steps, max_steps=1000,
                             diversity_weight=config['diversity_weight'])
        steps += 1000
        avg_loss, accuracy = evaluate(encoder_model, decoder_model, eval_data,
                                      criterion=criterion)
    wandb.finish()
if __name__ == "__main__":
    main()  # Call the main function
