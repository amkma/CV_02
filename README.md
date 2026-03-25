# CV_02: Edge & Boundary Detection System

**Team number:** 12  
**Team members:** * Ahmed Salah Geoshy Elshenawy  
* Abdullah Mohammad Khalifa  
* Mohamed Elsayed Attallah  

---

## Executive Summary

This project presents a high-performance, full-stack computer vision system focused on edge detection, shape recognition, and object segmentation. Developed to demonstrate a deep understanding of core algorithmic principles, the system bypasses standard black-box libraries (like OpenCV's built-in functions) in favor of a **First-Principles C++ Engine**. It features custom implementations of the Canny Edge Detector, Hough Transforms, and an Active Contour Model (Greedy Snake), all seamlessly integrated into a modern Django-based web interface via `pybind11`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Installation](#4-installation)
5. [Usage Instructions](#5-usage-instructions)
6. [Technical Specifications](#6-technical-specifications)
7. [File Descriptions](#7-file-descriptions)
8. [Acknowledgments](#8-acknowledgments)

---

## 1. Project Overview

### Background
Edge detection and boundary segmentation are foundational tasks in computer vision, serving as the critical first step for object recognition, medical image analysis, and autonomous tracking. Techniques range from parametric shape detection (Hough Transform) to energy-minimizing splines that dynamically adapt to object boundaries (Active Contours / Snakes).

### Problem Statement
While modern libraries provide high-level APIs for these tasks, they abstract away the mathematical and physical mechanics of image processing. Furthermore, implementing iterative optimization algorithms (like Snakes) in high-level languages like Python introduces severe performance bottlenecks, making interactive web-based segmentation sluggish and computationally expensive.

### Solution
This project implements a complete "digital pipeline" from the browser to the CPU. It strictly isolates the heavy mathematical computations in an optimized **C++17 Core Engine**, while maintaining a flexible Python/Django backend. The frontend provides a rich, interactive Canvas API allowing users to draw initial freehand contours, which are mathematically downsampled and passed to the C++ engine for high-speed gradient descent and Laplacian smoothing.

---

## 2. Key Features

### First-Principles Vision Engine (C++)
* **Custom Canny Edge Detection**: Implements a complete pipeline from scratch: Gaussian blurring, Sobel directional gradients, Non-Maximum Suppression (NMS), and BFS-based Hysteresis Thresholding.
* **Enhanced Hough Transforms**: 
  * *Lines*: Extracts exact segments using line-projection and optimized accumulator voting.
  * *Circles*: Utilizes gradient-direction voting to drastically reduce the 3D parameter space.
  * *Ellipses*: Implements anti-parallel gradient pairing and moment-based ellipse fitting.

### Active Contour Model (Greedy Snake)
* **Energy Minimization**: Dynamically fits a closed contour to an object by balancing Image Energy (edges) and Internal Energy (elasticity and curvature).
* **Distance Transform Navigation**: Uses a Chamfer 2-pass distance map rather than raw gradients, dramatically increasing the "capture range" (magnetic pull) of the object's edges.
* **Geometric Extraction**: Automatically generates 8-connectivity Freeman Chain Codes and calculates real-world metrics like Perimeter and Area (using the Shoelace formula).

### Interactive Web Dashboard (GUI)
* **Freehand Canvas Integration**: Users can sketch rough boundaries directly onto images via the browser.
* **Real-Time Downsampling**: The frontend JavaScript calculates cumulative arc length to interpolate and reduce thousands of raw mouse coordinates into exactly 60 equidistant anchor points.
* **Parametric Sliders**: Real-time control over Canny thresholds, Hough voting bins, and Snake energy weights (Alpha, Beta, Gamma).

---

## 3. System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Graphical User Interface (GUI)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ HTML5 Canvas │  │ Control Panel│  │ Diagnostic Dashboard │   │
│  │ (Freehand)   │  │ (JS Sliders) │  │ (Images & Metrics)   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────▲───────────┘   │
└─────────│─────────────────│─────────────────────│───────────────┘
          │ (JSON/Points)   │ (FormData)          │ (JSON Results)
┌─────────▼─────────────────▼─────────────────────┴───────────────┐
│              Django Backend (Python / views.py)                 │
│  - Receives AJAX requests, handles File I/O (Uploads)           │
│  - Parses JSON points and routes data via pybind11 bridge       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (std::vector & cv::Mat)
┌───────────────────────────▼─────────────────────────────────────┐
│              C++ Core Vision Engine (cv_core)                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Pre-processing & Fields                       │    │
│  │  - Custom Canny Edge & Distance Transform Map           │    │
│  │  - Sobel Gradients (dx, dy) generation                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Phase 2: Optimization Loop (Snake)                     │    │
│  │  - Gradient Descent (moves points towards dP/dx)        │    │
│  │  - Laplacian Smoothing (maintains spatial continuity)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

4. Installation
Prerequisites
 * Python 3.12+
 * CMake 3.15 or higher
 * OpenCV 4.x (C++ binaries installed and added to PATH)
 * A modern C++17 Compiler (e.g., MSVC v142/v143, GCC, or Clang)
Installation Steps
 * Clone the repository to your local machine.
 * Set up the virtual environment & install Python dependencies:
   python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

 * Build the C++ Core Engine:
   Ensure OpenCV_DIR is correctly mapped in your environment variables, then run the pybind11 build script:
   python build_cv_core.py build_ext --inplace

   (This will generate the cv_core.pyd or cv_core.so dynamic library).
 * Run the Django Server:
   python manage.py runserver

5. Usage Instructions
Running a Detection Task
 * Upload an Image: Open the application in your browser (http://localhost:8000) and upload a target image using the left-hand panel.
 * Task 1: Hough Transform:
   * Adjust the Canny thresholds and geometric voting parameters (rho, theta, min radius).
   * Click "Detect Shapes" to generate the 6-panel output showing isolated lines, circles, ellipses, and combined geometries.
 * Task 2: Active Contours (Snake):
   * Click "✏️ Draw Contour".
   * Click and drag your mouse around the object of interest to form a rough boundary.
   * Adjust the energy weights:
     * Alpha (\alpha): Controls the continuity and elasticity of the contour.
     * Beta (\beta): Controls the curvature and stiffness/smoothness.
     * Gamma (\gamma): Controls the image energy (magnetic pull towards the object's edges).
   * Click "Run Snake" to execute the C++ optimization. The final segmented object, along with its Area, Perimeter, and Chain Code, will be displayed.
6. Technical Specifications
Snake Energy Functional
The active contour minimizes the total energy functional. In this implementation, the optimization is handled via a greedy approach:
E_snake = ∫ (α * E_cont + β * E_curv + γ * E_img) ds

 * E_img is optimized via Gradient Descent against a Chamfer 2-pass distance transform, ensuring a wide capture range.
 * E_cont + E_curv is approximated efficiently using Laplacian Smoothing (averaging neighboring points with dynamic anchoring).
Geometric Feature Extraction
The area enclosed by the active contour is calculated rigorously using the Shoelace Formula over the final converged polynomial points:
Area = 0.5 * | Σ (x_i * y_{i+1} - x_{i+1} * y_i) |

7. File Descriptions
 * cv_custom_algorithms.h: The mathematical foundation. Contains from-scratch C++ implementations of Gaussian Blur, Sobel, Canny, Hough Transforms (Lines/Circles/Ellipses), and Distance Transforms.
 * cv_core.cpp: The primary engine and pybind11 wrapper. Houses the 2-phase Greedy Snake algorithm and handles data conversion between Python types and C++ STL/OpenCV structures.
 * views.py: The Django controllers. Responsible for handling HTTP requests, decoding JSON geometry, managing file I/O, and routing variables to the C++ engine.
 * app.js: The interactive frontend brain. Handles the Canvas API, mouse tracking, and features a crucial downsampleContour function that uses linear interpolation to compress raw freehand sketches into 60 equidistant nodes.
 * build_cv_core.py: A setuptools build script utilizing Pybind11Extension to locate OpenCV headers/libraries and compile the C++ source into a Python-importable module.
8. Acknowledgments
Institution: Cairo University, Faculty of Engineering
Department: Systems & Biomedical Engineering
Course: Computer Vision (CV_02)
(This project demonstrates low-level systems programming and high-level web integration for academic engineering purposes).

