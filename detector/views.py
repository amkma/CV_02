"""Views for the detector app."""

import os
import uuid
from django.conf import settings
from django.shortcuts import render, redirect

from .forms import HoughForm, SnakeForm

# Import the C++ module — built with build_cv_core.py
try:
    import cv_core
except ImportError:
    cv_core = None


def _save_upload(f):
    """Save an uploaded file to MEDIA_ROOT/uploads/ and return its abs path."""
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    name = f"{uuid.uuid4().hex}_{f.name}"
    path = os.path.join(upload_dir, name)
    with open(path, "wb") as dest:
        for chunk in f.chunks():
            dest.write(chunk)
    return path


def _media_url(abs_path):
    """Convert an absolute path under MEDIA_ROOT to a URL."""
    rel = os.path.relpath(abs_path, settings.MEDIA_ROOT)
    return settings.MEDIA_URL + rel.replace("\\", "/")


# ── 1. Upload page (home) ──────────────────────────────────────────

def upload(request):
    return render(request, "detector/upload.html", {
        "hough_form": HoughForm(),
        "snake_form": SnakeForm(),
        "cv_core_loaded": cv_core is not None,
    })


# ── 2. Hough detection ─────────────────────────────────────────────

def hough_detect(request):
    if request.method != "POST":
        return redirect("upload")

    form = HoughForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "detector/upload.html", {
            "hough_form": form,
            "snake_form": SnakeForm(),
            "errors": form.errors,
        })

    if cv_core is None:
        return render(request, "detector/upload.html", {
            "hough_form": form,
            "snake_form": SnakeForm(),
            "errors": {"cv_core": "C++ module not compiled. Run: python build_cv_core.py build_ext --inplace"},
        })

    img_path = _save_upload(request.FILES["image"])
    results_dir = os.path.join(settings.MEDIA_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    prefix = os.path.join(results_dir, uuid.uuid4().hex)

    d = form.cleaned_data
    result = cv_core.detect_all_shapes(
        img_path, prefix,
        d["canny_t1"], d["canny_t2"], 3,
        1.0, 1.0, d["line_thresh"], d["min_line_len"], d["max_line_gap"],
        d["dp"], d["min_dist"], d["circle_p1"], d["circle_p2"],
        d["min_r"], d["max_r"],
        20,
    )

    context = {
        "original":      _media_url(img_path),
        "edges":         _media_url(result["edges"]),
        "lines":         _media_url(result["lines"]),
        "circles":       _media_url(result["circles"]),
        "ellipses":      _media_url(result["ellipses"]),
        "combined":      _media_url(result["combined"]),
        "line_count":    result["line_count"],
        "circle_count":  result["circle_count"],
        "ellipse_count": result["ellipse_count"],
    }
    return render(request, "detector/hough_result.html", context)


# ── 3. Snake detection ─────────────────────────────────────────────

def snake_detect(request):
    if request.method != "POST":
        return redirect("upload")

    form = SnakeForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "detector/upload.html", {
            "hough_form": HoughForm(),
            "snake_form": form,
            "errors": form.errors,
        })

    if cv_core is None:
        return render(request, "detector/upload.html", {
            "hough_form": HoughForm(),
            "snake_form": form,
            "errors": {"cv_core": "C++ module not compiled. Run: python build_cv_core.py build_ext --inplace"},
        })

    img_path = _save_upload(request.FILES["image"])
    results_dir = os.path.join(settings.MEDIA_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{uuid.uuid4().hex}_snake.png")

    d = form.cleaned_data
    result = cv_core.active_contour_greedy(
        img_path, out_path,
        d["center_x"], d["center_y"], d["radius"],
        d["num_points"],
        d["alpha"], d["beta"], d["gamma"],
        d["window"], d["iterations"],
    )

    context = {
        "original":   _media_url(img_path),
        "contour":    _media_url(result["output"]),
        "chain_code": result["chain_code"],
        "perimeter":  f'{result["perimeter"]:.2f}',
        "area":       f'{result["area"]:.2f}',
        "num_points": result["num_points"],
    }
    return render(request, "detector/snake_result.html", context)
