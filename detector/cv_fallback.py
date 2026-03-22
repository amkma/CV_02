"""
Pure-Python fallback for cv_core C++ module.

Uses OpenCV (cv2) to replicate the same API so the Django app works
without compiling the C++ pybind11 extension.
"""

import os
import math
import cv2
import numpy as np


def _ensure_parent_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def canny_edge(input_path, output_path, threshold1=50.0, threshold2=150.0, aperture=3):
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {input_path}")
    ap = max(3, aperture if aperture % 2 == 1 else aperture + 1)
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=ap)
    _ensure_parent_dir(output_path)
    cv2.imwrite(output_path, edges)
    return output_path


def hough_lines(input_path, edge_path, output_path,
                rho=1.0, theta_deg=1.0, threshold=50,
                min_line_length=50.0, max_line_gap=10.0):
    color = cv2.imread(input_path, cv2.IMREAD_COLOR)
    edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if color is None or edges is None:
        raise RuntimeError("Could not read image(s)")

    theta = theta_deg * math.pi / 180.0
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    canvas = color.copy()
    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    _ensure_parent_dir(output_path)
    cv2.imwrite(output_path, canvas)
    return {"output": output_path, "line_count": line_count}


def hough_circles(input_path, output_path,
                  dp=1.2, min_dist=30.0,
                  param1=100.0, param2=30.0,
                  min_radius=0, max_radius=0):
    color = cv2.imread(input_path, cv2.IMREAD_COLOR)
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if color is None or gray is None:
        raise RuntimeError("Could not read image(s)")

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp, min_dist, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    canvas = color.copy()
    circle_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_count = len(circles[0])
        for c in circles[0]:
            center = (int(c[0]), int(c[1]))
            radius = int(c[2])
            cv2.circle(canvas, center, radius, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 3, (0, 255, 0), -1)

    _ensure_parent_dir(output_path)
    cv2.imwrite(output_path, canvas)
    return {"output": output_path, "circle_count": circle_count}


def detect_ellipses(input_path, edge_path, output_path, min_contour_points=20):
    color = cv2.imread(input_path, cv2.IMREAD_COLOR)
    edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if color is None or edges is None:
        raise RuntimeError("Could not read image(s)")

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    canvas = color.copy()
    count = 0
    for c in contours:
        if len(c) < max(5, min_contour_points):
            continue
        ell = cv2.fitEllipse(c)
        w, h = ell[1]
        ratio = w / (h + 1e-6)
        if ratio < 0.1 or ratio > 10.0:
            continue
        cv2.ellipse(canvas, ell, (255, 0, 255), 2, cv2.LINE_AA)
        count += 1

    _ensure_parent_dir(output_path)
    cv2.imwrite(output_path, canvas)
    return {"output": output_path, "ellipse_count": count}


def detect_all_shapes(input_path, output_prefix,
                      canny_t1=50.0, canny_t2=150.0, canny_aperture=3,
                      rho=1.0, theta_deg=1.0, line_threshold=50,
                      min_line_length=50.0, max_line_gap=10.0,
                      dp=1.2, min_dist=30.0,
                      circle_param1=100.0, circle_param2=30.0,
                      min_radius=0, max_radius=0,
                      min_contour_points=20):
    # 1. Canny
    edge_p = output_prefix + "_edges.png"
    canny_edge(input_path, edge_p, canny_t1, canny_t2, canny_aperture)

    # 2. Lines
    line_p = output_prefix + "_lines.png"
    ld = hough_lines(input_path, edge_p, line_p,
                     rho, theta_deg, line_threshold,
                     min_line_length, max_line_gap)

    # 3. Circles
    circ_p = output_prefix + "_circles.png"
    cd = hough_circles(input_path, circ_p,
                       dp, min_dist, circle_param1, circle_param2,
                       min_radius, max_radius)

    # 4. Ellipses
    ell_p = output_prefix + "_ellipses.png"
    ed = detect_ellipses(input_path, edge_p, ell_p, min_contour_points)

    # 5. Combined overlay
    color = cv2.imread(input_path, cv2.IMREAD_COLOR)
    edges = cv2.imread(edge_p, cv2.IMREAD_GRAYSCALE)
    canvas = color.copy()

    theta = theta_deg * math.pi / 180.0

    # Draw lines
    lines = cv2.HoughLinesP(edges, rho, theta, line_threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    # Draw circles
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp, min_dist, param1=circle_param1, param2=circle_param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles_r = np.uint16(np.around(circles))
        for c in circles_r[0]:
            cv2.circle(canvas, (int(c[0]), int(c[1])), int(c[2]),
                       (0, 255, 0), 2, cv2.LINE_AA)

    # Draw ellipses
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if len(c) < max(5, min_contour_points):
            continue
        ell = cv2.fitEllipse(c)
        w, h = ell[1]
        ratio = w / (h + 1e-6)
        if ratio < 0.1 or ratio > 10.0:
            continue
        cv2.ellipse(canvas, ell, (255, 0, 255), 2, cv2.LINE_AA)

    comb_p = output_prefix + "_combined.png"
    _ensure_parent_dir(comb_p)
    cv2.imwrite(comb_p, canvas)

    return {
        "edges": edge_p,
        "lines": line_p,
        "circles": circ_p,
        "ellipses": ell_p,
        "combined": comb_p,
        "line_count": ld["line_count"],
        "circle_count": cd["circle_count"],
        "ellipse_count": ed["ellipse_count"],
    }


def active_contour_greedy(input_path, output_path,
                          center_x, center_y, radius,
                          num_points=60,
                          alpha=1.0, beta=1.0, gamma=1.5,
                          window_size=7, iterations=100):
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    # Gradient magnitude for image energy
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    cv2.normalize(grad_mag, grad_mag, 0.0, 1.0, cv2.NORM_MINMAX)

    # Initialise contour as circle
    n = max(num_points, 8)
    snake = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        snake.append([center_x + radius * math.cos(angle),
                      center_y + radius * math.sin(angle)])
    snake = np.array(snake, dtype=np.float64)

    half_w = window_size // 2

    # Greedy iterations
    for _ in range(iterations):
        moved = False

        # Average distance for continuity
        d_avg = 0.0
        for i in range(n):
            prev = (i - 1 + n) % n
            d_avg += np.linalg.norm(snake[i] - snake[prev])
        d_avg /= n

        for i in range(n):
            prev_idx = (i - 1 + n) % n
            next_idx = (i + 1) % n

            best_energy = 1e18
            best_pos = snake[i].copy()

            for dy in range(-half_w, half_w + 1):
                for dx in range(-half_w, half_w + 1):
                    cx = snake[i][0] + dx
                    cy = snake[i][1] + dy

                    if cx < 1 or cy < 1 or cx >= gray.shape[1] - 1 or cy >= gray.shape[0] - 1:
                        continue

                    candidate = np.array([cx, cy])

                    # E_continuity
                    dist = np.linalg.norm(candidate - snake[prev_idx])
                    e_cont = (d_avg - dist) ** 2

                    # E_curvature
                    curv = snake[prev_idx] - 2.0 * candidate + snake[next_idx]
                    e_curv = curv[0] ** 2 + curv[1] ** 2

                    # E_image
                    ix = int(round(cx))
                    iy = int(round(cy))
                    g = grad_mag[iy, ix]
                    e_img = -(g * g)

                    energy = alpha * e_cont + beta * e_curv + gamma * e_img

                    if energy < best_energy:
                        best_energy = energy
                        best_pos = candidate.copy()

            if not np.array_equal(best_pos, snake[i]):
                snake[i] = best_pos
                moved = True

        if not moved:
            break

    # Chain code (8-connectivity Freeman)
    def direction(frm, to):
        dx = int(round(to[0] - frm[0]))
        dy = int(round(to[1] - frm[1]))
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        mapping = {
            (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
            (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7,
        }
        return mapping.get((dx, dy), -1)

    chain_code = []
    for i in range(n):
        nxt = (i + 1) % n
        code = direction(snake[i], snake[nxt])
        if code >= 0:
            chain_code.append(code)

    # Perimeter from chain code
    perimeter = sum(1.0 if c % 2 == 0 else math.sqrt(2.0) for c in chain_code)

    # Area via shoelace
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += snake[i][0] * snake[j][1]
        area -= snake[j][0] * snake[i][1]
    area = abs(area) * 0.5

    # Draw final contour
    canvas = color.copy()
    pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in snake], dtype=np.int32)
    cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(canvas, tuple(p), 3, (0, 0, 255), -1)

    _ensure_parent_dir(output_path)
    cv2.imwrite(output_path, canvas)

    return {
        "output": output_path,
        "chain_code": " ".join(str(c) for c in chain_code),
        "perimeter": perimeter,
        "area": area,
        "num_points": n,
    }
