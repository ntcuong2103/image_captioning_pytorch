## Image captioning

- Adapt from https://www.kaggle.com/code/mdteach/image-captioning-with-attention-pytorch
- Spacy

```
pip install spacy
```

- English tokenizer download
```
python -m spacy download en_core_web_sm
```

### Train

```
python trainer.py
```
### Test
```
python test.py
python eval.py
```

### Current version

- Trainer tested ok!
- Added metrics (bleu-1 in validation), saved best models with metrics
- Added evaluation after generating captions.

### Next
- Add beam search
