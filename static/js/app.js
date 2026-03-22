/**
 * app.js — Single-page AJAX controller for CV_02
 *
 * Handles: image upload preview, slider sync, debounced AJAX calls,
 *          result image + stats updates.
 */

const API = {
    upload: "/api/upload/",
    hough:  "/api/hough/",
    snake:  "/api/snake/",
};

let imagePath = null;   // server-side abs path of uploaded image
let imageUrl  = null;   // browser-accessible URL of uploaded image

// ── DOM refs ──

const imageInput   = document.getElementById("image-input");
const uploadText   = document.getElementById("upload-text");
const previewThumb = document.getElementById("preview-thumb");
const placeholder  = document.getElementById("placeholder");
const btnHough     = document.getElementById("btn-hough");
const btnSnake     = document.getElementById("btn-snake");


const houghSection = document.getElementById("hough-results");
const snakeSection = document.getElementById("snake-results");

// ── Slider sync ──

document.querySelectorAll(".ctrl").forEach(ctrl => {
    const range = ctrl.querySelector('input[type="range"]');
    const val   = ctrl.querySelector(".val");
    const step  = parseFloat(range.step);

    const update = () => {
        const v = parseFloat(range.value);
        val.textContent = step < 1 ? v.toFixed(1) : v;
        const pct = ((v - range.min) / (range.max - range.min)) * 100;
        range.style.setProperty("--fill", pct + "%");
    };

    range.addEventListener("input", update);
    update(); // initial fill
});

// ── Image upload ──

imageInput.addEventListener("change", async () => {
    const file = imageInput.files[0];
    if (!file) return;

    // Local preview
    const reader = new FileReader();
    reader.onload = e => {
        previewThumb.src = e.target.result;
        previewThumb.hidden = false;
    };
    reader.readAsDataURL(file);
    uploadText.textContent = file.name.length > 20
        ? file.name.slice(0, 17) + "…"
        : file.name;

    // Upload to server
    const fd = new FormData();
    fd.append("image", file);

    try {
        const res = await fetch(API.upload, { method: "POST", body: fd });
        const data = await res.json();
        imagePath = data.path;
        imageUrl  = data.url;
        placeholder.hidden = true;
        btnHough.disabled = false;
        btnSnake.disabled = false;
    } catch (err) {
        console.error("Upload failed:", err);
    }
});

// ── Gather slider values ──

function gatherParams(containerId) {
    const params = {};
    document.querySelectorAll(`#${containerId} .ctrl`).forEach(ctrl => {
        const key   = ctrl.dataset.key;
        const range = ctrl.querySelector('input[type="range"]');
        params[key] = range.value;
    });
    return params;
}

// ── AJAX helpers ──

async function runHough() {
    if (!imagePath) return;


    const params = gatherParams("hough-controls");
    params.image_path = imagePath;

    const fd = new FormData();
    for (const [k, v] of Object.entries(params)) fd.append(k, v);

    try {
        const res  = await fetch(API.hough, { method: "POST", body: fd });
        const data = await res.json();

        if (data.error) { console.error(data.error); return; }

        document.getElementById("img-original").src = imageUrl;
        document.getElementById("img-edges").src    = data.edges;
        document.getElementById("img-lines").src    = data.lines;
        document.getElementById("img-circles").src  = data.circles;
        document.getElementById("img-ellipses").src = data.ellipses;
        document.getElementById("img-combined").src = data.combined;

        document.getElementById("hough-stats").innerHTML =
            `<span class="stat"><strong>Lines:</strong> ${data.line_count}</span>` +
            `<span class="stat"><strong>Circles:</strong> ${data.circle_count}</span>` +
            `<span class="stat"><strong>Ellipses:</strong> ${data.ellipse_count}</span>`;

        placeholder.hidden = true;
        snakeSection.hidden = true;
        houghSection.hidden = false;

    } catch (err) {
        console.error("Hough failed:", err);

    }
}

async function runSnake() {
    if (!imagePath) return;


    const params = gatherParams("snake-controls");
    params.image_path = imagePath;

    const fd = new FormData();
    for (const [k, v] of Object.entries(params)) fd.append(k, v);

    try {
        const res  = await fetch(API.snake, { method: "POST", body: fd });
        const data = await res.json();

        if (data.error) { console.error(data.error); return; }

        document.getElementById("img-snake-orig").src = imageUrl;
        document.getElementById("img-contour").src    = data.contour;

        document.getElementById("snake-stats").innerHTML =
            `<span class="stat"><strong>Points:</strong> ${data.num_points}</span>` +
            `<span class="stat"><strong>Perimeter:</strong> ${data.perimeter} px</span>` +
            `<span class="stat"><strong>Area:</strong> ${data.area} px²</span>`;

        const ccBox  = document.getElementById("chain-code-box");
        const ccText = document.getElementById("chain-code-text");
        ccText.textContent = data.chain_code;
        ccBox.hidden = false;

        placeholder.hidden = true;
        houghSection.hidden = true;
        snakeSection.hidden = false;

    } catch (err) {
        console.error("Snake failed:", err);

    }
}



// ── Button handlers ──

btnHough.addEventListener("click", runHough);
btnSnake.addEventListener("click", runSnake);
