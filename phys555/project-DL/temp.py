import h5py
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


# The Dataset class provides basic input/output data pickup
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, indices, input_label, output_label):
        super(torch.utils.data.Dataset, self).__init__()
        self.data_path = data_path
        self.indices = indices
        self.input_label = input_label
        self.output_label = output_label

    def __getitem__(self, index):
        with h5py.File(self.data_path, 'r') as f:
            x = f[self.input_label][index]
            y = f[self.output_label][index]

            if len(y.shape) == 2:
                y = y.reshape(-1, y.shape[0], y.shape[1])

        x = torch.as_tensor(x, device='cpu', dtype=torch.float32)
        y = torch.as_tensor(y, device='cpu', dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.indices)


def get_train_valid_loader(data_path,
                           batch_size,
                           num_train,
                           valid_size=0.1,
                           shuffle=False,
                           num_workers=1,
                           val_data_path=None,
                           pin_memory=False,
                           input_label='cubes',
                           output_label='outs'
                           ):
    """
    Creates the data loaders for the training and validation datasets.

    Parameters
    ----------
    data_path:  str
        Path/directory to the training dataset.
    batch_size: int
        Batch size during training
    valid_size: float, optional
        The fraction (0-1) of the reference set to be used for validation (the fraction for training is 1 - valid_size)
    shuffle: boolean, optional
        Whether to shuffle the training set indices after every epoch
    num_workers: int, optional
        How many workers to load data in parallel
    val_data_path: str, optional
        Path/directory to the validation dataset (if different from the training set)
    pin_memory: boolean, optional
        If you load your samples in the Dataset on CPU and would like to push it during training to the GPU,
        you can speed up the host to device transfer by enabling pin_memory.
    input_label: str, optional
        Label in the h5 file corresponding to the input data
    output_label: str, optional
        Label in the h5 file corresponding to the output (target) data
    Returns
    -------
    train_loader: :class:`torch.DataLoader`
        A Dataloader for the training set.
    valid_loader: :class:`torch.DataLoader`
        A Dataloader for the validation set.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    frac_train = 1 - valid_size
    indices_reference = np.arange(num_train)
    indices_train = sorted(indices_reference[:int(frac_train * len(indices_reference))])

    # If you didn't specify a separate path for the validation data, use the regular data file path.
    # Otherwise, use the validation data file path
    if not val_data_path:
        val_data_path = data_path
        indices_val = sorted(indices_reference[int(frac_train * len(indices_reference)):])
    else:
        val_data_path = h5py.File(val_data_path, 'r')
        indices_val = np.arange(int(frac_train * len(indices_reference)))

    # Initialize data loaders
    print('Initializing data loaders...')
    train_dataset = Dataset(data_path,
                            indices=indices_train,
                            input_label=input_label,
                            output_label=output_label)
    valid_dataset = Dataset(val_data_path,
                            indices=indices_val,
                            input_label=input_label,
                            output_label=output_label)

    trainset = torch.utils.data.Subset(train_dataset, indices_train)
    validset = torch.utils.data.Subset(valid_dataset, indices_val)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle
    )
    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=False
    )

    return train_loader, valid_loader


# Here's how to use the dataloaders:

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

######### DataLoaders ###########
train_loader, val_loader = get_train_valid_loader(data_path='/path/to/data.h5',
                                                  num_train=500,
                                                  batch_size=16,
                                                  num_workers=1,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  input_label='cubes',
                                                  output_label='outs')


for epoch in range(100):

    ######### Training Loop ###########
    for i, data in enumerate(tqdm(train_loader), 0):
        input = data[0].to(device)
        target = data[1].to(device)

        # Rest of training code

    ######### Validation Loop ###########
    for i, data in enumerate(tqdm(val_loader), 0):
        input = data[0].to(device)
        target = data[1].to(device)

        with torch.no_grad():
            # Rest of validation code
