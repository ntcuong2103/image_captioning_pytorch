from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data import ImageCaptioningData, Vocabulary
from image_caption_model_lt import ImageCaptionModelTrainer

if __name__ == "__main__":
    dm = ImageCaptioningData(batch_size=1, workers=1)
    dm.setup()

    model = ImageCaptionModelTrainer(
        vocab=Vocabulary("vocab/Flickr8k_freq5.txt"),
    ).load_from_checkpoint("checkpoint/flickr8k-attention/lightning_logs/version_1/checkpoints/epoch=10-val_loss=2.7895.ckpt")

    trainer = Trainer(
        enable_checkpointing=False,
        fast_dev_run=False,
        gpus=1,
    )

    trainer.test(model, dm)
