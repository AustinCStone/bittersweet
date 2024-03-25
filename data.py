import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


def bytes_to_bit_chunks(bytestring, chunk_size=128):
    """Yield bit chunks of a given size from a bytestring."""
    bit_stream = ''.join(format(byte, '08b') for byte in bytestring)
    
    # Yield chunks of size chunk_size
    for i in range(0, len(bit_stream), 1):
        return bit_stream[i:i+chunk_size]


class ShakespeareBitsDataset(Dataset):
    def __init__(self, text_path, chunk_size=128, train=True, split_percentage=0.8):
        self.chunk_size = chunk_size
        self.train = train
        # Load the text file
        with open(text_path, 'br') as file:
            text = file.read()

        
        # Convert text to binary representation
        self.binary_str = ''.join(format(byte, '08b') for byte in text)
        
        # Calculate the total number of complete chunks
        total_samples = len(self.binary_str)
        
        # Determine the split index
        split_index = int(total_samples * split_percentage)
        
        # Generate indices for train and test sets
        self.indices = list(range(total_samples))
        if self.train:
            self.indices = self.indices[:split_index]
        else:
            self.indices = self.indices[split_index:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Adjust index based on subset
        true_idx = self.indices[idx] 
        # Extract a chunk of the specified size
        start_idx = true_idx
        if start_idx + self.chunk_size > len(self.binary_str):
            start_idx = len(self.binary_str) - self.chunk_size 
        end_idx = start_idx + self.chunk_size
        sample = self.binary_str[start_idx:end_idx]
        
        # Convert the string of bits to a tensor of integers (0s and 1s)
        tensor = torch.tensor([int(bit) for bit in sample], dtype=torch.int32)
        
        return tensor


def create_data_loaders(text_path, chunk_size=128, split_percentage=0.8, batch_size=10):
    train_dataset = ShakespeareBitsDataset(text_path, chunk_size=chunk_size, train=True, split_percentage=split_percentage)
    test_dataset = ShakespeareBitsDataset(text_path, chunk_size=chunk_size, train=False, split_percentage=split_percentage)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_loader, test_loader


# def load2():
#     # Example usage
#     text_path = 'input.txt'
#     chunk_size = 128  # Or any other size you want to specify

#     # Create the DataLoader instances
#     train_loader, test_loader = create_data_loaders(text_path, chunk_size=chunk_size, batch_size=1)

#     train_data = []
#     for batch in train_loader:
#         train_data.extend(batch)

#     test_data = []
#     for batch in test_loader:
#         test_data.extend(batch)

#     return train_data, test_data

# def load():
#     # Preprocess data
#     with open('input.txt', 'br') as file:
#         data = file.read()
    
#     size = len(data)
#     split_line = int(0.8*size)
#     train = data[:split_line]
#     test = data[split_line:]

#     return bytes_to_bit_chunks(train), bytes_to_bit_chunks(test)