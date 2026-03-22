"""Views for the detector app."""

import os
import uuid

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# ── Make sure OpenCV DLLs are findable ──
_opencv_bin = r"C:\Program Files\opencv\build\x64\vc16\bin"
if os.path.isdir(_opencv_bin):
    os.add_dll_directory(_opencv_bin)
    if _opencv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _opencv_bin + ";" + os.environ.get("PATH", "")

import cv_core


# ── Helpers ────────────────────────────────────────────────────────

def _save_upload(f):
    """Save uploaded file, return absolute path."""
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(f.name)[1].lower() or ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(upload_dir, name)
    with open(path, "wb") as dest:
        for chunk in f.chunks():
            dest.write(chunk)
    return path


def _media_url(abs_path):
    """Absolute path → media URL."""
    rel = os.path.relpath(abs_path, settings.MEDIA_ROOT)
    return settings.MEDIA_URL + rel.replace("\\", "/")


# ── Single page ───────────────────────────────────────────────────

def home(request):
    return render(request, "detector/home.html")


# ── API: upload image ─────────────────────────────────────────────

@csrf_exempt
def api_upload(request):
    if request.method != "POST" or "image" not in request.FILES:
        return JsonResponse({"error": "POST with image required"}, status=400)
    path = _save_upload(request.FILES["image"])
    return JsonResponse({"path": path, "url": _media_url(path)})


# ── API: Hough detection ──────────────────────────────────────────

@csrf_exempt
def api_hough(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    results_dir = os.path.join(settings.MEDIA_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    prefix = os.path.join(results_dir, uuid.uuid4().hex)

    result = cv_core.detect_all_shapes(
        img_path, prefix,
        float(request.POST.get("canny_t1", 50)),
        float(request.POST.get("canny_t2", 150)),
        3,
        float(request.POST.get("rho", 1.0)),
        float(request.POST.get("theta_deg", 1.0)),
        int(request.POST.get("line_thresh", 50)),
        float(request.POST.get("min_line_len", 50)),
        float(request.POST.get("max_line_gap", 10)),
        float(request.POST.get("dp", 1.2)),
        float(request.POST.get("min_dist", 30)),
        float(request.POST.get("circle_p1", 100)),
        float(request.POST.get("circle_p2", 30)),
        int(request.POST.get("min_r", 0)),
        int(request.POST.get("max_r", 0)),
        int(request.POST.get("min_contour_pts", 20)),
    )

    return JsonResponse({
        "edges":         _media_url(result["edges"]),
        "lines":         _media_url(result["lines"]),
        "circles":       _media_url(result["circles"]),
        "ellipses":      _media_url(result["ellipses"]),
        "combined":      _media_url(result["combined"]),
        "line_count":    result["line_count"],
        "circle_count":  result["circle_count"],
        "ellipse_count": result["ellipse_count"],
    })


# ── API: Snake detection ──────────────────────────────────────────

@csrf_exempt
def api_snake(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    results_dir = os.path.join(settings.MEDIA_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{uuid.uuid4().hex}_snake.png")

    result = cv_core.active_contour_greedy(
        img_path, out_path,
        int(request.POST.get("center_x", 200)),
        int(request.POST.get("center_y", 200)),
        int(request.POST.get("radius", 100)),
        int(request.POST.get("num_points", 60)),
        float(request.POST.get("alpha", 1.0)),
        float(request.POST.get("beta", 1.0)),
        float(request.POST.get("gamma", 1.5)),
        int(request.POST.get("window", 7)),
        int(request.POST.get("iterations", 100)),
    )

    return JsonResponse({
        "contour":    _media_url(result["output"]),
        "chain_code": result["chain_code"],
        "perimeter":  round(result["perimeter"], 2),
        "area":       round(result["area"], 2),
        "num_points": result["num_points"],
    })
