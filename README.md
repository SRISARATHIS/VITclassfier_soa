# Hard Cat vs Dog — ViT

Fine-tune a pretrained Vision Transformer for difficult cat/dog examples, then
use `molab_dashboard.py` in a Molab marimo session to train on a GPU and run
interactive image inference.

## Cute web classifier

`docs/` is a responsive, static upload website for GitHub Pages. It runs a
CLIP Vision Transformer in the visitor's browser, so images are not uploaded
to a server. The first visit downloads the browser model; later visits use the
browser cache. Pushing changes under `docs/` deploys the site through the
included GitHub Actions workflow.

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

## Scratch-built ViT on Kaggle Cats-vs-Dogs

`train_scratch_vit.py` contains the Vision Transformer architecture itself:
patch projection, a learnable class token and position embeddings, eight
Transformer encoder blocks, and a classification head. It uses no pretrained
weights. The Molab job downloads Kaggle's public
`shaunthesheep/microsoft-catsvsdogs-dataset` archive with `curl`, trains this
model on the GPU, and saves `artifacts/scratch_vit/best.pt`.

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
