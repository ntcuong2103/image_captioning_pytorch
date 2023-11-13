import os
from collections import Counter
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import spacy
import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

spacy_eng = spacy.load("en_core_web_sm")


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """

    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets


class Vocabulary:
    def __init__(self, vocab_file) -> None:
        self.load_vocab(vocab_file)

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

        frequencies = Counter()
        idx = len(self.itos)

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def save_vocab(self, fn):
        with open(fn, "wt") as f:
            f.writelines([w + "\n" for w in self.itos.values()])

    def load_vocab(self, fn):
        self.itos = {idx: line.strip() for idx, line in enumerate(open(fn).readlines())}
        self.stoi = {v: k for k, v in self.itos.items()}

    def numericalize(self, text):
        """For each word in the text corresponding index token for that word form the vocab built as list"""
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """

    def __init__(self, root_dir, captions_file, vocab_file, ids_file=None, transforms=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        if ids_file != None:
            ids = [line.strip() for line in open(ids_file).readlines()]
            self.df = self.df[self.df.image.isin(ids)]

        self.transforms = transforms

        self.imgs = self.df["image"].tolist()
        self.captions = self.df["caption"].tolist()

        self.vocab = Vocabulary(vocab_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transfromation to the image
        if self.transforms is not None:
            img = self.transforms(img)

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)


class ImageCaptioningData(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 10,
        workers: int = 5,
        images_dir="../image_captioning/Images",
        train_ids: str = "../image_captioning/Flickr8k_text/Flickr_8k.trainImages.txt",
        val_ids: str = "../image_captioning/Flickr8k_text/Flickr_8k.devImages.txt",
        test_ids: str = "../image_captioning/Flickr8k_text/Flickr_8k.testImages.txt",
        captions_file: str = "../image_captioning/Flickr8k_text/captions.txt",
        vocab_file: str = "vocab/Flickr8k_freq5.txt",
        img_size: int = 512,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.images_dir = images_dir
        self.captions_file = captions_file
        self.vocab_file = vocab_file
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.img_size = img_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()])

            self.train_dataset = FlickrDataset(
                root_dir=self.images_dir,
                captions_file=self.captions_file,
                vocab_file=self.vocab_file,
                ids_file=self.train_ids,
                transforms=transforms,
            )
            self.val_dataset = FlickrDataset(
                root_dir=self.images_dir,
                captions_file=self.captions_file,
                vocab_file=self.vocab_file,
                ids_file=self.val_ids,
                transforms=transforms,
            )
        if stage == "test" or stage is None:
            self.test_dataset = FlickrDataset(
                root_dir=self.images_dir,
                captions_file=self.captions_file,
                vocab_file=self.vocab_file,
                ids_file=self.test_ids,
                transforms=transforms,
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=CapsCollate(
                pad_idx=self.train_dataset.vocab.stoi["<PAD>"], batch_first=True
            ),
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=CapsCollate(
                pad_idx=self.train_dataset.vocab.stoi["<PAD>"], batch_first=True
            ),
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    pass

    ## build and save vocab
    # captions_file='/home/tuancuong/workspaces/image_captioning/Flickr8k_text/captions.txt'
    # freq_threshold = 5
    # df = pd.read_csv(captions_file)
    # vocab = Vocabulary(freq_threshold)
    # vocab.build_vocab(pd.read_csv(captions_file)["caption"].tolist())
    # os.makedirs('vocab', exist_ok=True)
    # vocab.save_vocab(f'vocab/Flickr8k_freq{freq_threshold}.txt')
    # exit(0)

    # dataset = FlickrDataset(root_dir='/home/tuancuong/workspaces/image_captioning/Images',
    #                         captions_file='/home/tuancuong/workspaces/image_captioning/Flickr8k_text/captions.txt',
    #                         vocab_file='vocab/Flickr8k_freq5.txt',
    #                         ids_file='/home/tuancuong/workspaces/image_captioning/Flickr8k_text/Flickr_8k.devImages.txt')

    # dataset.__getitem__(0)
    # exit(0)

    # dm = ImageCaptioningData()
    # dm.setup()
    # trainloader = dm.train_dataloader()
    # for img, label in trainloader:
    #     print()
    #     pass
