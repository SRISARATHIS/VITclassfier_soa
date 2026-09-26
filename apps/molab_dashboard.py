"""Run this file with `marimo edit apps/molab_dashboard.py` in Molab."""
import marimo

app = marimo.App(width="full")


@app.cell
def __():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    import marimo as mo
    return Path, json, mo, os, subprocess, sys, time


@app.cell
def __(mo):
    mo.md("""
    # 🐾 Hard Cases Lab
    ### Vision Transformer training and cat/dog inference on Molab GPU
    Sync this project from GitHub, start a GPU job, watch validation metrics,
    and drop in any image to test the finished model.
    """)
    return


@app.cell
def __(mo):
    repo_url = mo.ui.text(label="GitHub repository URL", placeholder="https://github.com/you/catdog_class_vit.git")
    data_url = mo.ui.text(label="Optional dataset .zip URL", placeholder="https://…/hard-cats-dogs.zip")
    prepare = mo.ui.run_button(label="Prepare workspace", kind="success")
    mo.vstack([repo_url, data_url, prepare])
    return data_url, prepare, repo_url


@app.cell
def __(Path, data_url, mo, prepare, repo_url, subprocess):
    workspace = Path("/tmp/catdog_vit_workspace")
    prepare_message = "Paste a GitHub repository URL to begin."
    if prepare.value:
        if not repo_url.value.strip():
            prepare_message = "A GitHub repository URL is required."
        else:
            subprocess.run(["rm", "-rf", str(workspace)], check=False)
            clone = subprocess.run(["git", "clone", "--depth", "1", repo_url.value.strip(), str(workspace)], capture_output=True, text=True)
            if clone.returncode:
                prepare_message = f"Clone failed: {clone.stderr[-500:]}"
            else:
                install = subprocess.run(["bash", "-lc", f"cd {workspace} && {__import__('sys').executable} -m pip install -q -e ."], capture_output=True, text=True)
                if install.returncode:
                    prepare_message = f"Dependency installation failed: {install.stderr[-500:]}"
                elif data_url.value.strip():
                    archive = workspace / "dataset.zip"
                    fetch = subprocess.run(["curl", "-L", "--fail", "--progress-bar", data_url.value.strip(), "-o", str(archive)], capture_output=True, text=True)
                    unpack = subprocess.run(["unzip", "-oq", str(archive), "-d", str(workspace / "data")], capture_output=True, text=True) if not fetch.returncode else fetch
                    prepare_message = "Workspace and dataset are ready." if not unpack.returncode else f"Dataset download failed: {fetch.stderr[-500:]}"
                else:
                    prepare_message = "Workspace is ready. Add data/train/{cat,dog} and data/val/{cat,dog}, then start training."
    mo.callout(prepare_message, kind="info")
    return workspace,


@app.cell
def __(mo):
    epochs = mo.ui.slider(4, 40, value=20, label="Epochs")
    batch_size = mo.ui.dropdown(options=[16, 32, 64, 128], value=64, label="Batch size")
    start = mo.ui.run_button(label="Start GPU training", kind="warn")
    mo.hstack([epochs, batch_size, start], justify="start", gap=2)
    return batch_size, epochs, start


@app.cell
def __(batch_size, epochs, mo, start, subprocess, sys, workspace):
    job = getattr(__import__('builtins'), "_catdog_job", None)
    if start.value and workspace.exists() and (workspace / "data" / "kaggle" / "PetImages").exists():
        artifacts = workspace / "artifacts"
        artifacts.mkdir(exist_ok=True)
        job = subprocess.Popen([sys.executable, "-m", "catdog_vit.train_scratch", "--data-root", "data/kaggle/PetImages", "--output-dir", "artifacts/scratch_vit",
                                "--epochs", str(epochs.value), "--batch-size", str(batch_size.value)],
                               cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        __import__('builtins')._catdog_job = job
    state = "No job started."
    if job is not None:
        state = "Training is running on the Molab GPU." if job.poll() is None else f"Job finished with code {job.returncode}."
    mo.callout(state, kind="warn" if "running" in state else "info")
    return job,


@app.cell
def __(Path, json, mo, time, workspace):
    refresh = mo.ui.refresh(options=[3, 5, 10], default_interval=5)
    metrics_path = workspace / "artifacts" / "scratch_vit" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"state": "waiting"}
    accuracy = metrics.get("val_accuracy", metrics.get("best_accuracy", 0))
    dashboard = mo.vstack([
        refresh,
        mo.md(f"### Job status: `{metrics.get('state', 'waiting')}`"),
        mo.stat(label="Best validation accuracy", value=f"{accuracy:.1%}" if accuracy else "—"),
        mo.md(f"**GPU/device:** {metrics.get('device', 'awaiting job')} · **Last update:** {time.strftime('%H:%M:%S', time.localtime(metrics.get('updated_at', time.time())))}"),
    ])
    dashboard
    return metrics,


@app.cell
def __(mo, workspace):
    checkpoint = workspace / "artifacts" / "scratch_vit" / "best.pt"
    history = workspace / "artifacts" / "scratch_vit" / "history.csv"
    if checkpoint.exists():
        mo.vstack([
            mo.md("### Download results via the notebook connection"),
            mo.download(data=checkpoint.read_bytes(), filename="catdog_vit_best.pt", label="Download best checkpoint"),
            mo.download(data=history.read_bytes(), filename="training_history.csv", label="Download training history") if history.exists() else mo.md("History will appear after epoch 1."),
        ])
    else:
        mo.md("Results will appear here when the first checkpoint is saved.")
    return checkpoint,


@app.cell
def __(mo):
    upload = mo.ui.file(label="Upload an image for inference", filetypes=[".jpg", ".jpeg", ".png", ".webp"])
    upload
    return upload,


@app.cell
def __(Path, checkpoint, mo, upload):
    if upload.value and checkpoint.exists():
        import io
        import torch
        import timm
        from PIL import Image
        from torchvision import transforms
        payload = upload.value[0]
        image = Image.open(io.BytesIO(payload.contents)).convert("RGB")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["model"] == "scratch_vit_384_8_6":
            from catdog_vit.train_scratch import VisionTransformer
            model = VisionTransformer()
        else:
            model = timm.create_model(saved["model"], pretrained=False, num_classes=len(saved["classes"]))
        model.load_state_dict(saved["state_dict"])
        model.eval()
        tf = transforms.Compose([transforms.Resize(int(saved["image_size"] * 1.15)), transforms.CenterCrop(saved["image_size"]), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        with torch.inference_mode():
            probability = torch.softmax(model(tf(image).unsqueeze(0)), dim=1)[0]
        index = probability.argmax().item()
        mo.vstack([image, mo.callout(f"Prediction: **{saved['classes'][index].upper()}** · confidence **{probability[index]:.1%}**", kind="success")])
    elif upload.value:
        mo.callout("Train a model first; no checkpoint is available yet.", kind="warn")
    return


if __name__ == "__main__":
    app.run()
