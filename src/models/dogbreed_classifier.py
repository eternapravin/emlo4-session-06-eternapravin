import lightning as L
import timm
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy

class DogBreedClassifier(L.LightningModule):
    def __init__(self, base_model: str, num_classes: int = 10, lr: float = 1e-3, pretrained: bool = True):
        super().__init__()
        self.lr = lr
        self.base_model = base_model
        self.num_classes = num_classes

        # Load pre-trained ResNet18 model
        self.model = timm.create_model(base_model, pretrained, num_classes=self.num_classes)

        # Multi-class accuracy with num_classes=2
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.validation_step_outputs = []

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
