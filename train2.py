# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers==0.0.23post1 mwparserfromhell datasets fast-pytorch-kmeans
# wandb key: 4d89c43f67fc55f37cc6e65e9304ef29b1a454f3
import torch
import numpy as np
import time
import data
import torch.nn.functional as F
import modeling2 as modeling
import tqdm
from sklearn.cluster import MiniBatchKMeans
from fast_pytorch_kmeans import KMeans
# from torchviz import make_dot
import wandb
import os


DEBUG=True
USE_WANDB=False


def sample(model: torch.nn.Module, input_len: int, initial_seq: torch.Tensor, num_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
    model.eval()
    input_len = min(input_len, model.pos_encoder.pe.size(0) - 1)

    if initial_seq is None or initial_seq.size(0) == 0:
        raise ValueError("'initial_seq' must be provided and non-empty.")

    generated_seq = initial_seq.clone()

    with torch.no_grad():
        for _ in range(input_len - initial_seq.size(0)):
            output = model(generated_seq)
            logits = output[-1, 0] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=num_samples)
            generated_seq = torch.cat((generated_seq, next_token.unsqueeze(0)), dim=0)

    return generated_seq.squeeze()


def load_model(checkpoint_dir, model, model_name="encoder"):
    """
    Load the model from the latest checkpoint.
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(model_name) and f.endswith(".pt")]
    if checkpoints:
        # Sort files by their step number
        checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        model.load_state_dict(torch.load(latest_checkpoint))
        step_number = int(checkpoints[-1].split('_')[-1].split('.')[0])
        print(f"Restored {model_name} from {latest_checkpoint}")
        return model, step_number
    else:
        print(f"No checkpoints found for {model_name} in {checkpoint_dir}. Starting from scratch.")
        return model, 0


def manage_checkpoints(checkpoint_dir, max_checkpoints=5):
    # Get all checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    
    # If there are more than `max_checkpoints` files, remove the oldest
    if len(checkpoints) > max_checkpoints:
        # Sort files by their creation time
        checkpoints.sort(key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
        # Remove the oldest
        for f in checkpoints[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, f))
            print(f"Removed old checkpoint: {f}")


def save_model(encoder_model, decoder_model, checkpoint_dir, step_number):
    encoder_path = os.path.join(checkpoint_dir, f"encoder_model_step_{step_number}.pt")
    decoder_path = os.path.join(checkpoint_dir, f"decoder_model_step_{step_number}.pt")

    torch.save(encoder_model.state_dict(), encoder_path)
    torch.save(decoder_model.state_dict(), decoder_path)
    print(f"Saved models at step {step_number} to {checkpoint_dir}")


def evaluate(encoder_model, middle_model, decoder_model, eval_data, criterion,
             num_evals=1, print_predictions=True, samples_to_print=1):
    encoder_model.eval()  # Turn on evaluation mode
    decoder_model.eval()  # Turn on evaluation mode
    middle_model.eval() # Turn on evaluation mode

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for batch_idx, (batch, targets) in enumerate(eval_data):
            if batch_idx >= num_evals:  # Ensure it breaks at num_evals
                break
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = batch.to(device)
            T = batch.shape[1]  # Assuming T is the sequence length from inputs
            soft_preds = encoder_model(batch)
            soft_preds = middle_model(soft_preds)
            predictions = decoder_model(soft_preds)

            num_classes = predictions.shape[-1]
            reconstructed_flat = predictions.view(-1, num_classes)
            targets_flat = targets.view(-1).long()

            loss = criterion(reconstructed_flat, targets_flat)
            total_loss += loss.item()
            hard_predictions = torch.argmax(predictions, axis=-1)
            correct_predictions += (hard_predictions == targets).sum().item()
            total_predictions += hard_predictions.numel()

    avg_loss = total_loss / min(batch_idx + 1, num_evals)
    accuracy = correct_predictions / total_predictions * 100
    if USE_WANDB:
        wandb.log({'eval_loss': avg_loss, 'eval_accuracy': accuracy})
    print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy


def train(encoder_model, decoder_model, middle_model, train_data, criterion,
          optimizer, log_interval=1, max_steps=100, start_step=0):
    encoder_model.train()  # turn on train mode
    decoder_model.train()  # turn on train mode
    middle_model.train()

    criterion = torch.nn.CrossEntropyLoss()
    losses = {
        'cross_entropy_loss': [],
    }
    for batch_idx, (batch, targets) in enumerate(train_data):
        if batch_idx + start_step > max_steps:
            break
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch = batch.to(device)
        # Converting binary tokens into vectors.
        # Input from batch is 0s and 1s of shape [batch_size, T]
        # Output shape should be [batch_size, T, d_model]
        optimizer.zero_grad()

        soft_preds = encoder_model(batch) 
        soft_preds = middle_model(soft_preds)
        predictions = decoder_model(soft_preds)
        num_classes = predictions.shape[-1]
        loss = criterion(predictions.view(-1, num_classes), targets.view(-1).long())
        # TODO add loss weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder_model.parameters()) + list(decoder_model.parameters()), 0.5)
        optimizer.step()
        accuracy = (torch.argmax(predictions, axis=-1) == targets).sum().item() / targets.numel()
        print("Accuracy: ", accuracy)
        if batch_idx % log_interval == 0:  # log_interval could be, e.g., 10
            print(f'Batch: {batch_idx + start_step}, Loss: {loss.item()}, ')
            if batch_idx % 100 == 0:
                predictions_flat = predictions.view(-1, num_classes)
                _, predicted_labels = torch.max(predictions_flat, 1)
                predicted_labels_reshaped = predicted_labels.view(batch.shape[0], batch.shape[1])
                print("Ground truth:", batch[0])
                print("Prediction:", predicted_labels_reshaped[0])
            losses['cross_entropy_loss'].append(loss.item())
            if USE_WANDB:
                wandb.log({'cross_entropy_loss': loss.item(), 'accuracy': accuracy})
    return {k: np.mean(v) for k, v in losses.items()}

def main():
    if DEBUG:
        config = {
            # Load data
            'chunk_size': 1024, # Encode 8 bytes sequence length.
            'split_percentage':0.8, # Use 80% of data for training.
            'batch_size': 32,
            # model hypers
            'lr':1e-3,
            # Encoder / decoder shared params
            'ntokens': 256,  # All bytes.
            'encoder_d_model': 320,
            'encoder_d_hid': 512,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
            'encoder_nlayers': 4,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            'encoder_nhead': 4,  # number of heads in ``nn.MultiheadAttention``
            'encoder_dropout': 0.2,  # dropout probability
            'compression_factor': 8,
            # Middle model shared params
            'd_model': 320,
            'd_hid': 512,
            'nlayers': 4,
            'nhead': 4,
            'dropout': .2,
            'eval_every': 100,
            'version': 'shakespeare',
            'restore_dir': None,
        }
    else:
        raise NotImplementedError()
    # start a new wandb run to track this script
    if USE_WANDB:
        wandb.init(
            # set the wandb project where this run will be logged
            project="rackitten-end-to-end",
            # track hyperparameters and run metadata
            config=config
        )
        run_id = wandb.run.id
        print("Run id is: ", run_id)
    else:
        run_id = "local_run"
    train_data, eval_data = data.create_data_loaders(
        chunk_size=config['chunk_size'],
        split_percentage=config['split_percentage'],
        batch_size=config['batch_size'],
        use_bits=False,
        version=config['version'],
        with_targets=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert config['nlayers'] % 2 == 0
    encoder_model = modeling.PoolExpandTransformerModel(
        ntoken=config['ntokens'],
        d_model=config['encoder_d_model'],
        nhead=config['encoder_nhead'],
        d_hid=config['encoder_d_hid'],
        nlayers_pre=config['encoder_nlayers'] // 2,
        nlayers_post=config['encoder_nlayers'] // 2,
        dropout=config['encoder_dropout'],
        include_linear=False,
        max_len=config['chunk_size'],
        compression_factor=config['compression_factor']).to(device)
    assert config['d_model'] % config['compression_factor'] == 0
    decoder_model = modeling.PoolExpandTransformerModel(
        ntoken=config['ntokens'],
        d_model=config['encoder_d_model'],
        d_hid=config['encoder_d_hid'],
        nlayers_pre=config['encoder_nlayers'] // 2,
        nlayers_post=config['encoder_nlayers'] // 2,
        nhead=config['encoder_nhead'],
        dropout=config['encoder_dropout'],
        include_linear=True,
        vector_input=True,
        max_len=config['chunk_size'],
        compression_factor=1./config["compression_factor"]).to(device)
    middle_model = modeling.PoolExpandTransformerModel(
        ntoken=-1, # vector input
        d_model=config['d_model'],
        d_hid=config['d_hid'],
        nlayers_pre=config['nlayers'] // 2,
        nlayers_post=config['nlayers'] // 2,
        nhead=config['nhead'],
        dropout=config['dropout'],
        include_linear=False,
        vector_input=True,
        max_len=config['chunk_size'] // config['compression_factor'],
        compression_factor=1).to(device)
    # Penalize the model for reconstructing the binary input.
    criterion = torch.nn.CrossEntropyLoss()
    # TODO: Optimizer params not persisted or restored. This causes loss
    # spikes on restoration.
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=config['lr'])
    checkpoint_dir = f'/tmp/{run_id}_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    steps = 0

    if config['restore_dir']:
        encoder_model, steps = load_model(config['restore_dir'], encoder_model, model_name="encoder")
        decoder_model, dec_steps = load_model(config['restore_dir'], decoder_model, model_name="decoder")
        middle_model, middle_steps = load_model(config['restore_dir'], middle_model, model_name='middle')
        assert steps == dec_steps
        print(f"Restored continuous model from {config['restore_dir']} at step {steps}")
    for _ in range(100): # Pretrain continuous
        train_losses = train(encoder_model, decoder_model, middle_model,
                             train_data,
                             criterion=criterion, optimizer=optimizer,
                             start_step=steps, max_steps=steps + config['eval_every'])
        steps += config['eval_every']
        avg_loss, accuracy = evaluate(encoder_model, decoder_model, middle_model,
                                      eval_data, criterion=criterion)
        print("Saving model....")
        save_model(encoder_model, decoder_model, middle_model, checkpoint_dir, steps)
        # manage_checkpoints(discrete_checkpoint_dir)  
    wandb.finish()
if __name__ == "__main__":
    main()  # Call the main function
