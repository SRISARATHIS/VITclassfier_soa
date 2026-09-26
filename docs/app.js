import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

env.allowLocalModels = false;
// Deployed as a fully static GitHub Pages app.
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const idle = document.querySelector("#idle-state");
const loading = document.querySelector("#loading-state");
const result = document.querySelector("#result-state");
const title = document.querySelector("#loading-title");
const copy = document.querySelector("#loading-copy");
const preview = document.querySelector("#preview");
let classifier;

function show(state) {
  idle.classList.toggle("hidden", state !== "idle");
  loading.classList.toggle("hidden", state !== "loading");
  result.classList.toggle("hidden", state !== "result");
}

async function model() {
  if (!classifier) {
    title.textContent = "Teaching my tiny brain…";
    copy.textContent = "Downloading the ViT model (first visit only)";
    classifier = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32", {
      // WASM is the broadest browser-compatible runtime for GitHub Pages.
      device: "wasm",
      progress_callback: (event) => {
        if (event.status === "progress") copy.textContent = `Loading model… ${Math.round(event.progress ?? 0)}%`;
      },
    });
  }
  return classifier;
}

async function classify(file) {
  if (!file || !file.type.startsWith("image/")) return;
  show("loading");
  preview.src = URL.createObjectURL(file);
  try {
    const predict = await model();
    title.textContent = "Sniffing the evidence…";
    copy.textContent = "Looking at whiskers, ears, and floof";
    const scores = await predict(preview.src, ["a photo of a cat", "a photo of a dog"]);
    const winner = scores[0];
    const animal = winner.label.includes("cat") ? "Cat! 🐱" : "Dog! 🐶";
    document.querySelector("#prediction").textContent = animal;
    document.querySelector("#confidence").textContent = `${Math.round(winner.score * 100)}% sure`;
    document.querySelector("#confidence-fill").style.width = `${Math.max(winner.score * 100, 4)}%`;
    document.querySelector("#detail").textContent = winner.score < .68 ? "A wonderfully tricky one — I’m making my best fuzzy guess." : "The ViT found some pretty convincing fuzzy clues.";
    show("result");
  } catch (error) {
    title.textContent = "Oops, my whiskers got tangled";
    copy.textContent = "Please check your connection and try again.";
    console.error(error);
  }
}

input.addEventListener("change", () => classify(input.files[0]));
["dragenter", "dragover"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => classify(event.dataTransfer.files[0]));
document.querySelector("#try-again").addEventListener("click", () => { input.value = ""; show("idle"); });
