import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
import torchvision.models as models 
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from src.utils_seq import create_dataset
from src.utils import read_data, compute_metrics
class ResidualCNN(pl.LightningModule):   
    """Pytorch Lightning instance of the residual cnn model applied on the protein sequences
    """
    def __init__(self,data, learning_rate = 3e-4, batch_size = 512):
        super(ResidualCNN, self).__init__()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.data=data
        self.y_true = []
        self.y_pred = []
        self.names = []
        self.define_parameters()

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        output_dim = max(data['train'].label) + 1
        self.embeddings = nn.Embedding(self.max_nb_chars, self.embedding_dim, padding_idx=0)
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3,bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=output_dim, bias=True)
        self.criterion = nn.CrossEntropyLoss()      

    # Defining the forward pass    
    def forward(self, data):
        x = self.embeddings(data)
        x = x.permute(0, 2, 1).unsqueeze(1)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        names, sequences, y = batch
        y_hat = self(sequences)
        loss = self.criterion(y_hat, y.view(-1))
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss
    
    def training_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True, batch_size=self.batch_size)
    
    def validation_step(self, batch, batch_nb):
        names, sequences, y = batch
        y_hat = self(sequences)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(preds,  y.view(-1))
        loss = self.criterion(y_hat, y.view(-1))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=self.batch_size)
    
    def test_step(self, batch, batch_nb):
        names, sequences, y = batch
        y_hat = self(sequences)
        preds = torch.argmax(y_hat, dim=1)
        self.y_true += y.detach().tolist()
        self.y_pred += preds.detach().tolist()
        self.names += list(names)
        self.test_accuracy.update(preds, y.view(-1))
        loss = self.criterion(y_hat, y.view(-1))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("test_acc", self.test_accuracy, prog_bar=True, batch_size=self.batch_size)
    
    def test_epoch_end(self, outputs):
        return compute_metrics(self.y_true, self.y_pred, self.path_results, 'CNN', names=self.names)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def prepare_data(self):
        self.X_train, self.X_val, self.X_test = create_dataset(self.data,max_nb_chars=self.max_nb_chars,
                                                max_sequence_length=self.max_sequence_length)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self.X_train
            self.val_dataset = self.X_val
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = self.X_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
    
    def define_parameters(self):
        self.max_sequence_length = 512
        self.max_nb_chars = 21
        self.embedding_dim=64
        self.path_results='results/'
  
def main(args):
    seed_everything(0)
    epochs = args.n_epochs
    data = read_data()
    net = net = ResidualCNN(data, args.learning_rate, args.batch_size)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/CNN/", save_top_k=2, monitor="val_acc", mode='max')
    trainer = pl.Trainer(accelerator='gpu',
            max_epochs=epochs,
            # limit_train_batches=1,
            # limit_val_batches=1,
            # limit_test_batches=1,
            accumulate_grad_batches=4,
            callbacks=[checkpoint_callback],
                            )
    trainer.fit(net)
    out = trainer.test(ckpt_path='best')
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4,
        help = "Learning Rate to use for the optimizer")
    parser.add_argument("-bs", "--batch_size", type=int,default=512, 
	    help = "Batch Size for training")
    parser.add_argument("--n_epochs", type=int,default=10, 
	help = "Number of epochs to train the model")
    
    main(parser.parse_args())

