import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class EmbeddingsCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_test_end(self, trainer, pl_module):
        trainer.logger.experiment.add_embedding(
            pl_module.test_embeddings,
            pl_module.test_targets,
            global_step=trainer.global_step)


class ANN(pl.LightningModule):
    def __init__(self, data_dir='./'):
        super().__init__()
        # Set our init args as class attributes
        self.data_dir = data_dir
        self.test_targets = []
        self.test_embeddings = torch.zeros((0, 100),
                                           dtype=torch.float32,
                                           device='cuda:0')
        self.test_predictions = []
        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 13 * 13, 100)
        self.fc2 = nn.Linear(100, self.num_classes)
        # Define PyTorch model

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.maxpool1(x))
        x = x.view(-1, 16 * 13 * 13)
        x = self.fc1(x)
        y = self.fc2(F.relu(x))
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        x, y = batch
        embeddings, logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.test_predictions.extend(preds.detach().cpu().tolist())
        self.test_targets.extend(y.detach().cpu().tolist())
        self.test_embeddings = torch.cat((self.test_embeddings, embeddings), 0)
        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit':
            dataset_full = MNIST(self.data_dir,
                                 train=True,
                                 transform=self.transform)
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            self.dataset_test = MNIST(self.data_dir,
                                      train=False,
                                      transform=self.transform)
            np.random.seed(19)
            random_indices = np.random.uniform(0, 10000, 100).astype(np.uint8)
            outlier_list = []
            for i in range(100):
                outlier = np.random.uniform(0, 255, (28, 28)).astype(np.uint8)
                outlier_list.append(outlier)
            for idx in range(len(random_indices)):
                self.dataset_test.data[random_indices[idx]] = torch.ByteTensor(
                    outlier_list[idx])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=32)


if __name__ == "__main__":
    model = ANN()
    embedding_callback = EmbeddingsCallback()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        save_weights_only=True)
    trainer = pl.Trainer(gpus=1,
                         max_epochs=5,
                         progress_bar_refresh_rate=20,
                         callbacks=[checkpoint_callback, embedding_callback])
    trainer.fit(model)
    trainer.test()
