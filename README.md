# Hard Cat vs Dog — ViT

Fine-tune a pretrained Vision Transformer for difficult cat/dog examples, then
use `molab_dashboard.py` in a Molab marimo session to train on a GPU and run
interactive image inference.

## Dataset layout

Put a deliberately difficult, curated dataset in a directory with this shape:

```
data/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
```

The model starts from ImageNet-pretrained `vit_small_patch16_224`. Strong
augmentation, label smoothing, MixUp, CutMix, AdamW, cosine decay, and early
stopping improve robustness for visually ambiguous images.

## Training

```bash
python train.py --data-dir data --output-dir artifacts --epochs 20 --batch-size 64
```

The best checkpoint is `artifacts/best.pt`; metrics are written incrementally
to `artifacts/metrics.json` and `artifacts/history.csv`, so the dashboard can
show progress while a job is running.

## Molab workflow

1. Create a GitHub repository, commit and push this project.
2. Open the paired Molab notebook. Paste the repository URL and an optional
   dataset URL, then click **Prepare workspace**.
3. Start training from the dashboard. It polls the metrics file and exposes
   downloads of the final checkpoint and training history through the notebook
   connection.
4. Upload any image after training to get a cat/dog prediction and confidence.

Private repositories need a URL that Molab can authenticate to (or an
appropriate Git credential configured in the Molab session).
