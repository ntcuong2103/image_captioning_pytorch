from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    preds = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in  open('predictions.txt').readlines()}
    gts = defaultdict(list)
    for line in open("../image_captioning/Flickr8k_text/captions.txt").readlines()[1:]:
        gts[line.strip().split(',')[0]] += [line.strip().split(',')[1]]

    references = [gts[k] for k in preds.keys()]
    hypotheses = [preds[k] for k in preds.keys()]

    print(
        "BLEU-1: %f" % corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    )
    print(
        "BLEU-2: %f" % corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    )
    print(
        "BLEU-3: %f"
        % corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    )
    print(
        "BLEU-4: %f"
        % corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    )