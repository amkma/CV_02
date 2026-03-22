/**
 * app.js — Single-page AJAX controller for CV_02
 *
 * Handles: image upload preview, slider sync, debounced AJAX calls,
 *          result image + stats updates, and interactive contour drawing.
 */

const API = {
    upload: "/api/upload/",
    hough:  "/api/hough/",
    snake:  "/api/snake/",
};

let imagePath = null;   // server-side abs path of uploaded image
let imageUrl  = null;   // browser-accessible URL of uploaded image

// ── Drawing state ──
let drawMode     = false;
let drawnPoints  = [];     // [{x, y}] in IMAGE coordinates
let contourClosed = false;

// ── DOM refs ──

const imageInput   = document.getElementById("image-input");
const uploadText   = document.getElementById("upload-text");
const previewThumb = document.getElementById("preview-thumb");
const placeholder  = document.getElementById("placeholder");
const btnHough     = document.getElementById("btn-hough");
const btnSnake     = document.getElementById("btn-snake");
const btnDraw      = document.getElementById("btn-draw");
const btnClear     = document.getElementById("btn-clear");
const drawHint     = document.getElementById("draw-hint");
const drawCount    = document.getElementById("draw-count");
const pointCount   = document.getElementById("point-count");
const drawCanvas   = document.getElementById("draw-canvas");

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
        btnDraw.disabled  = false;
        btnClear.disabled = false;
        // Clear any previous drawing
        clearDrawing();
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

// ── Drawing ──

function enterDrawMode() {
    drawMode = true;
    drawnPoints = [];
    contourClosed = false;
    btnDraw.classList.add("active");
    drawHint.hidden = false;
    drawCount.hidden = false;
    pointCount.textContent = "0";

    // Hide circle sliders when drawing
    document.querySelectorAll(".circle-ctrl").forEach(el => el.style.display = "none");

    // Show snake section with original image for drawing
    if (imageUrl) {
        document.getElementById("img-snake-orig").src = imageUrl;
        placeholder.hidden = true;
        houghSection.hidden = true;
        snakeSection.hidden = false;
        setupCanvas();
    }
}

function exitDrawMode() {
    drawMode = false;
    btnDraw.classList.remove("active");
    drawHint.hidden = true;
    if (drawCanvas) drawCanvas.style.cursor = "";
}

function clearDrawing() {
    drawnPoints = [];
    contourClosed = false;
    pointCount.textContent = "0";
    drawCount.hidden = true;

    // Show circle sliders again
    document.querySelectorAll(".circle-ctrl").forEach(el => el.style.display = "");

    if (drawCanvas) {
        const ctx = drawCanvas.getContext("2d");
        ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    }
    exitDrawMode();
}

function setupCanvas() {
    const img = document.getElementById("img-snake-orig");

    const doSetup = () => {
        // Canvas covers the full figure area (same as CSS)
        const figRect = drawCanvas.parentElement.getBoundingClientRect();
        drawCanvas.width  = figRect.width;
        drawCanvas.height = figRect.height - 28; // minus figcaption
        drawCanvas.style.cursor = drawMode ? "crosshair" : "";
        redrawCanvas();
    };

    if (img.complete && img.naturalWidth > 0) {
        doSetup();
    } else {
        img.addEventListener("load", doSetup, { once: true });
    }
}

/**
 * Compute the actual rendered rectangle of an <img> with object-fit:contain
 * inside its container, relative to the container's top-left.
 */
function getRenderedImageRect() {
    const img = document.getElementById("img-snake-orig");
    const containerW = drawCanvas.width;
    const containerH = drawCanvas.height;
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;

    const scale = Math.min(containerW / natW, containerH / natH);
    const renderedW = natW * scale;
    const renderedH = natH * scale;
    const offsetX = (containerW - renderedW) / 2;
    const offsetY = (containerH - renderedH) / 2;

    return { offsetX, offsetY, renderedW, renderedH, scale, natW, natH };
}

function getImageCoords(e) {
    const canvasRect = drawCanvas.getBoundingClientRect();
    const { offsetX, offsetY, scale } = getRenderedImageRect();

    // Click position relative to the canvas element
    const cx = e.clientX - canvasRect.left;
    const cy = e.clientY - canvasRect.top;

    // Convert to image pixel coordinates
    return {
        x: Math.round((cx - offsetX) / scale),
        y: Math.round((cy - offsetY) / scale),
    };
}

function redrawCanvas() {
    if (!drawCanvas) return;
    const ctx = drawCanvas.getContext("2d");
    ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

    if (drawnPoints.length === 0) return;

    const { offsetX, offsetY, scale } = getRenderedImageRect();

    // Convert image coords → canvas coords
    const toCanvas = (p) => ({
        cx: offsetX + p.x * scale,
        cy: offsetY + p.y * scale,
    });

    // Draw lines between points
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const c0 = toCanvas(drawnPoints[0]);
    ctx.moveTo(c0.cx, c0.cy);
    for (let i = 1; i < drawnPoints.length; i++) {
        const ci = toCanvas(drawnPoints[i]);
        ctx.lineTo(ci.cx, ci.cy);
    }
    if (contourClosed) {
        ctx.closePath();
    }
    ctx.stroke();

    // Draw points
    drawnPoints.forEach((p, i) => {
        const cp = toCanvas(p);
        ctx.beginPath();
        ctx.arc(cp.cx, cp.cy, i === 0 ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? "#ff4d6a" : "#6c6cff";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}

// ── Freehand sketch helpers ──

let isDrawing = false;
let rawSketch = [];  // dense freehand points in image coords

/**
 * Downsample a dense polyline to ~targetN evenly-spaced points.
 */
function downsampleContour(pts, targetN) {
    if (pts.length <= targetN) return pts;

    // Compute cumulative arc length
    const cumLen = [0];
    for (let i = 1; i < pts.length; i++) {
        const dx = pts[i].x - pts[i - 1].x;
        const dy = pts[i].y - pts[i - 1].y;
        cumLen.push(cumLen[i - 1] + Math.sqrt(dx * dx + dy * dy));
    }
    const totalLen = cumLen[cumLen.length - 1];
    if (totalLen === 0) return pts.slice(0, targetN);

    const step = totalLen / targetN;
    const result = [pts[0]];
    let nextDist = step;
    let j = 1;

    while (result.length < targetN && j < pts.length) {
        if (cumLen[j] >= nextDist) {
            // Interpolate between pts[j-1] and pts[j]
            const segLen = cumLen[j] - cumLen[j - 1];
            const t = segLen > 0 ? (nextDist - cumLen[j - 1]) / segLen : 0;
            result.push({
                x: Math.round(pts[j - 1].x + t * (pts[j].x - pts[j - 1].x)),
                y: Math.round(pts[j - 1].y + t * (pts[j].y - pts[j - 1].y)),
            });
            nextDist += step;
        } else {
            j++;
        }
    }
    return result;
}

// Canvas freehand drawing handlers
if (drawCanvas) {
    drawCanvas.addEventListener("mousedown", (e) => {
        if (!drawMode || e.button !== 0) return;
        isDrawing = true;
        rawSketch = [];
        drawnPoints = [];
        contourClosed = false;
        const pt = getImageCoords(e);
        rawSketch.push(pt);
        redrawCanvas();
    });

    drawCanvas.addEventListener("mousemove", (e) => {
        if (!isDrawing) return;
        const pt = getImageCoords(e);

        // Only add if moved at least a few pixels from last point (image coords)
        const last = rawSketch[rawSketch.length - 1];
        const dist = Math.sqrt((pt.x - last.x) ** 2 + (pt.y - last.y) ** 2);
        if (dist < 3) return;

        rawSketch.push(pt);
        // Show raw sketch while drawing
        drawnPoints = rawSketch;
        pointCount.textContent = rawSketch.length;
        redrawCanvas();
    });

    drawCanvas.addEventListener("mouseup", (e) => {
        if (!isDrawing) return;
        isDrawing = false;

        if (rawSketch.length < 5) {
            // Too few points, discard
            rawSketch = [];
            drawnPoints = [];
            redrawCanvas();
            return;
        }

        // Downsample to ~60 evenly-spaced points and close
        drawnPoints = downsampleContour(rawSketch, 60);
        contourClosed = true;
        pointCount.textContent = drawnPoints.length;
        exitDrawMode();
        redrawCanvas();
    });

    // Prevent context menu on canvas
    drawCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
}

btnDraw.addEventListener("click", () => {
    if (drawMode) {
        exitDrawMode();
    } else {
        enterDrawMode();
    }
});

btnClear.addEventListener("click", clearDrawing);

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

    // If user drew points, send them instead of circle params
    if (drawnPoints.length >= 3) {
        params.points = JSON.stringify(drawnPoints.map(p => [p.x, p.y]));
        // Remove circle-specific params
        delete params.center_x;
        delete params.center_y;
        delete params.radius;
        delete params.num_points;
    }

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

        // Clear the drawing overlay after results come in
        if (drawCanvas) {
            const ctx = drawCanvas.getContext("2d");
            ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        }

    } catch (err) {
        console.error("Snake failed:", err);
    }
}



// ── Button handlers ──

btnHough.addEventListener("click", runHough);
btnSnake.addEventListener("click", runSnake);
