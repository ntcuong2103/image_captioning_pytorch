from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data import ImageCaptioningData, Vocabulary
from image_caption_model_lt import ImageCaptionModelTrainer

if __name__ == "__main__":
    dm = ImageCaptioningData(batch_size=10, workers=10)

    model = ImageCaptionModelTrainer(
        vocab=Vocabulary("vocab/Flickr8k_freq5.txt"),
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-4,
    )

    trainer = Trainer(
        checkpoint_callback=True,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename="{epoch}-{val_bleu_1:.4f}", save_top_k=5, monitor="val_bleu_1", mode="max"
            ),
        ],
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        default_root_dir="checkpoint/flickr8k-attention",
        deterministic=False,
        max_epochs=50,
        log_every_n_steps=50,
        gpus=1,
        amp_backend="apex",
        amp_level="O1",
        # precision=16,
        # strategy='ddp',
    )

    trainer.fit(model, dm)
