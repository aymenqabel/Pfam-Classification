import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import esm

from src.dataset import PfamDataset
from src.utils import compute_metrics, read_data
   
class EsmModel(nn.Module):
    """ESM Model to use with 8 million parameters
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        num_features = 320
        self.model.lm_head = nn.Identity(num_features)
        self.fc = nn.Linear(num_features, hidden_channels)
    
    def transform_to_batch(self, names, sequences):
        new_data = [(names[i], sequences[i]) for i in range(len(sequences))]
        _, _, batch_tokens = self.batch_converter(new_data)
        return batch_tokens.long().cuda()

    def forward(self, names, sequences):
        inputs = self.transform_to_batch(names, sequences)
        out = self.model(inputs)['logits']
        x = out[:, 0, :]
        return self.fc(x)



class ESM(pl.LightningModule):   
    """Pytorch Lightning Instance of the ESM model
    """
    def __init__(self, data, lr = 3e-4, batch_size = 8, ):
        super(ESM, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.path_results = 'results/'
        self.y_true = []
        self.y_pred = []
        self.names = []
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        output_dim = max(data['train'].label) + 1
        self.model =  EsmModel(output_dim)  
        self.data = data
        self.criterion = nn.CrossEntropyLoss() 

    def freeze_encoder(self):
        for param in self.model.model.parameters():
          param.requires_grad = False
          
    def forward(self, names, sequences):
        return self.model(names, sequences)
    
    def training_step(self, batch, batch_idx):
        names, sequences, y = batch
        y_hat = self(names, sequences)
        loss = self.criterion(y_hat, y.view(-1))
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True, batch_size=self.batch_size)
    
    def validation_step(self, batch, batch_nb):
        names, sequences, y = batch
        y_hat = self(names, sequences)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy.update(preds,  y.view(-1))
        loss = self.criterion(y_hat, y.view(-1))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=self.batch_size)
    
    def test_step(self, batch, batch_nb):
        names, sequences, y = batch
        y_hat = self(names, sequences)
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
        return compute_metrics(self.y_true, self.y_pred, self.path_results, 'ESM', names=self.names)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = PfamDataset(self.data['train']['sequence'].values,self.data['train']['sequence_name'].values,self.data['train']['label'].values)
            self.val_dataset = PfamDataset(self.data['val']['sequence'].values,self.data['val']['sequence_name'].values,self.data['val']['label'].values)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = PfamDataset(self.data['test']['sequence'].values,self.data['test']['sequence_name'].values,self.data['test']['label'].values)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

def main(args):
    seed_everything(0)
    epochs = args.n_epochs
    data = read_data()
    net = ESM(data, args.learning_rate, args.batch_size)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/ESM/", save_top_k=2, monitor="val_acc", mode='max')
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
    parser.add_argument("-bs", "--batch_size", type=int,default=8, 
	    help = "Batch Size for training")
    parser.add_argument("--n_epochs", type=int,default=10, 
	help = "Number of epochs to train the model")
    main(parser.parse_args())
    