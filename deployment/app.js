
const input = document.getElementById("imageInput");
const button = document.getElementById("predictBtn");
const status = document.getElementById("status");
const preview = document.getElementById("preview");
const results = document.getElementById("results");

let session;
let labels;
let config;
let selectedFile;

async function initialize() {
  try {
    status.textContent = "Loading trained CNN...";

    const [labelsResponse, configResponse] = await Promise.all([
      fetch("./labels.json"),
      fetch("./model_config.json")
    ]);

    if (!labelsResponse.ok || !configResponse.ok) {
      throw new Error("Could not load model metadata.");
    }

    labels = await labelsResponse.json();
    config = await configResponse.json();

    if (!Array.isArray(labels) || !Number.isInteger(config.image_size)) {
      throw new Error("Invalid labels or model configuration.");
    }

    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";

    session = await ort.InferenceSession.create(
      "./model.onnx",
      { executionProviders: ["wasm"] }
    );

    status.textContent = "CNN ready. Upload a leaf image.";
    button.disabled = !selectedFile;

  } catch (error) {
    status.textContent = "Model loading failed: " + error.message;
    console.error(error);
  }
}

input.addEventListener("change", () => {
  selectedFile = input.files[0];
  results.innerHTML = "";
  button.disabled = !selectedFile || !session;

  if (!selectedFile) {
    preview.style.display = "none";
    return;
  }

  if (!selectedFile.type.startsWith("image/") ||
      selectedFile.size > 10 * 1024 * 1024) {
    status.textContent = "Please select an image under 10 MB.";
    selectedFile = null;
    button.disabled = true;
    return;
  }

  const url = URL.createObjectURL(selectedFile);
  preview.onload = () => URL.revokeObjectURL(url);
  preview.onerror = () => URL.revokeObjectURL(url);
  preview.src = url;
  preview.style.display = "block";
});

async function preprocess(file) {
  const image = await createImageBitmap(file);
  const size = config.image_size;

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, size, size);
  image.close();

  const pixels = ctx.getImageData(0, 0, size, size).data;
  const area = size * size;
  const tensorData = new Float32Array(3 * area);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < area; i++) {
    for (let channel = 0; channel < 3; channel++) {
      let value = pixels[i * 4 + channel] / 255;

      if (config.normalize === true) {
        value = (value - mean[channel]) / std[channel];
      }

      tensorData[channel * area + i] = value;
    }
  }

  return new ort.Tensor(
    "float32",
    tensorData,
    [1, 3, size, size]
  );
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function showPredictions(probabilities) {
  results.replaceChildren();

  const top5 = probabilities
    .map((probability, index) => ({
      name: labels[index],
      probability
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);

  for (const prediction of top5) {
    const row = document.createElement("div");
    row.className = "result";

    const label = document.createElement("div");
    const name = prediction.name
      .replaceAll("___", " - ")
      .replaceAll("_", " ");

    label.textContent =
      `${name}: ${(prediction.probability * 100).toFixed(1)}%`;

    const bar = document.createElement("div");
    bar.className = "bar";

    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${prediction.probability * 100}%`;

    bar.appendChild(fill);
    row.append(label, bar);
    results.appendChild(row);
  }
}

button.addEventListener("click", async () => {
  if (!session || !selectedFile) return;

  button.disabled = true;
  status.textContent = "Analysing leaf...";

  try {
    const tensor = await preprocess(selectedFile);

    const feeds = {
      [session.inputNames[0]]: tensor
    };

    const output = await session.run(feeds);
    const logits = Array.from(
      output[session.outputNames[0]].data
    );

    if (logits.length !== labels.length) {
      throw new Error("Output size does not match class labels.");
    }

    const probabilities = softmax(logits);
    showPredictions(probabilities);

    status.textContent = "Prediction completed.";

  } catch (error) {
    status.textContent = "Prediction failed: " + error.message;
    console.error(error);

  } finally {
    button.disabled = false;
  }
});

initialize();