import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


def bytes_to_int_chunks(bytestring, chunk_size=128):
    """Yield integer chunks of a given size from a bytestring."""
    for i in range(0, len(bytestring), chunk_size):
        chunk = bytestring[i:i+chunk_size]
        yield torch.tensor([byte for byte in chunk], dtype=torch.uint8)


class BytesDataset(Dataset):
    def __init__(self, text_path, chunk_size=128, train=True, split_percentage=0.8, use_bits=False,
                 with_targets=False):
        print("Loading data...")
        self.chunk_size = chunk_size
        self.train = train
        self.use_bits = use_bits
        self.with_targets = with_targets
        if self.with_targets:
            self.chunk_size += 1

        # Load the text file
        with open(text_path, 'br') as file:
            text = file.read()

        if self.use_bits:
            # Convert text to binary representation
            self.binary_str = ''.join(format(byte, '08b') for byte in text)
            total_samples = len(self.binary_str)
        else:
            # Use the raw bytes
            self.bytes = [x for x in text]
            total_samples = len(self.bytes)

        # Determine the split index
        split_index = int(total_samples * split_percentage)

        # Generate indices for train and test sets
        self.indices = list(range(total_samples))
        if self.train:
            self.indices = self.indices[:split_index]
        else:
            self.indices = self.indices[split_index:]
        print("Done loading data...")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]

        if self.use_bits:
            # Extract a chunk of the specified size from the binary string
            start_idx = true_idx
            end_idx = start_idx + self.chunk_size
            sample = self.binary_str[start_idx:end_idx]

            # Pad the sample if it's shorter than the chunk size
            if len(sample) < self.chunk_size:
                sample += '0' * (self.chunk_size - len(sample))

            # Convert the string of bits to a tensor of integers (0s and 1s)
            tensor = torch.tensor([int(bit) for bit in sample], dtype=torch.int32)
        else:
            # Extract a chunk of the specified size from the bytes
            start_idx = true_idx
            end_idx = start_idx + self.chunk_size
            chunk = self.bytes[start_idx:end_idx]

            # Pad the chunk if it's shorter than the chunk size
            if len(chunk) < self.chunk_size:
                chunk += [0] * (self.chunk_size - len(chunk))

            # Convert the bytes to a tensor of integers (0-255)
            tensor = torch.tensor([byte for byte in chunk], dtype=torch.int32)

        if self.with_targets:
            return tensor[:-1], tensor[1:]

        return tensor

def create_data_loaders(version, chunk_size=128, split_percentage=0.8, batch_size=10, use_bits=False, with_targets=False):
    if version == 'shakespeare':
        text_path = 'input.txt'
    elif version == 'wiki':
        text_path = 'simple_wiki.txt'
    else:
        raise NotImplementedError(f'Version {version} not implemented.')
    train_dataset = BytesDataset(text_path, chunk_size=chunk_size, train=True, split_percentage=split_percentage, use_bits=use_bits,
                                 with_targets=with_targets)
    test_dataset = BytesDataset(text_path, chunk_size=chunk_size, train=False, split_percentage=split_percentage, use_bits=use_bits,
                                with_targets=with_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader


# x, y = create_data_loaders('input.txt', chunk_size=8 * 32, use_bits=False)
# for xx in x:
#     import pdb; pdb.set_trace()