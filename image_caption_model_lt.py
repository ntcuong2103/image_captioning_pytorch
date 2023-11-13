import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule

from encoder_decoder import EncoderDecoder as Model
from nltk.translate.bleu_score import corpus_bleu


class ImageCaptionModelTrainer(LightningModule):
    def __init__(
        self,
        vocab,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = vocab
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi["<PAD>"])

        # metrics
        self.model = Model(
            embed_size=300,
            vocab=vocab,
            attention_dim=256,
            encoder_dim=2048,
            decoder_dim=512,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, captions, _ = batch
        outputs, attentions = self.model(image, captions)
        targets = captions[:, 1:]
        loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.reshape(-1))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def eval_step(self, batch, batch_idx, prefix: str):
        image, captions, image_name = batch
        outputs, attentions = self.model.forward(image, captions, sequential=True)
        targets = captions[:, 1:]
        bleu_1 = corpus_bleu([[self.vocab.itos[i] for i in targets[0][:-1].cpu().numpy()]], [outputs[:-1]], weights=(1.0, 0, 0, 0))
        
        self.log(f"{prefix}_bleu_1", bleu_1, prog_bar=True, on_step=True, on_epoch=True)
        return image_name[0], ' '.join(outputs[:-1])

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")
    
    def test_epoch_end(self, test_outputs) -> None:
        with open('predictions.txt', 'w') as f:
            f.writelines([f'{fn}\t{output}\n' for fn, output in test_outputs])
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
        return [optimizer], [scheduler]
