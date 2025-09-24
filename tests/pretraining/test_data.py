import pytest
import torch
import os

from ebm_lib.pretraining.data import HeuristicPretrainingDataModule

@pytest.fixture
def data_module():
    """Provides an instance of the HeuristicPretrainingDataModule."""
    # Use a temporary directory for data
    return HeuristicPretrainingDataModule(data_dir="./test_data", batch_size=4)

def test_data_module_init(data_module):
    """Tests that the DataModule initializes correctly."""
    assert data_module.batch_size == 4
    assert os.path.exists("./test_data") is False or len(os.listdir("./test_data")) == 0

def test_data_module_prepare_data(data_module):
    """Tests that the data is downloaded."""
    data_module.prepare_data()
    # Check that the CIFAR10 data directory has been created
    assert os.path.exists(os.path.join(data_module.data_dir, "cifar-10-batches-py"))

def test_data_module_setup(data_module):
    """Tests that the setup method creates the datasets."""
    data_module.prepare_data() # Must be called before setup
    data_module.setup(stage='fit')

    assert hasattr(data_module, 'train_dataset')
    assert len(data_module.train_dataset) == 50000 # CIFAR10 training set size

    data_module.setup(stage='test')
    assert hasattr(data_module, 'test_dataset')
    assert len(data_module.test_dataset) == 10000 # CIFAR10 test set size

def test_data_module_dataloader(data_module):
    """
    Tests that the train_dataloader returns a batch of augmented images
    with the correct shape and type.
    """
    data_module.prepare_data()
    data_module.setup(stage='fit')
    dataloader = data_module.train_dataloader()

    # Get one batch from the dataloader
    batch = next(iter(dataloader))

    assert isinstance(batch, torch.Tensor)
    # Shape should be (batch_size, channels, height, width)
    # CIFAR10 is 3 channels, 32x32. Our augmentations keep this size.
    assert batch.shape == (data_module.batch_size, 3, 32, 32)
    # The ToTensorV2 and Normalize transforms should result in a float tensor
    assert batch.dtype == torch.float32

    # Clean up the downloaded data
    # Note: In a real CI environment, this might be handled by the runner.
    import shutil
    if os.path.exists(data_module.data_dir):
        shutil.rmtree(data_module.data_dir)
