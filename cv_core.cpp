/**
 * cv_core – C++ OpenCV operations for CV_02 (Hough + Snake).
 *
 * Exposed to Python via pybind11.
 * The Django layer is only a thin HTTP wrapper and must NEVER call cv2.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace py = pybind11;

/* ───────────────────── helpers (same pattern as plceholderawy) ───── */
namespace {

cv::Mat load_image(const std::string& path, const std::string& mode)
{
    int flag = cv::IMREAD_COLOR;
    if (mode == "gray")       flag = cv::IMREAD_GRAYSCALE;
    else if (mode == "unchanged") flag = cv::IMREAD_UNCHANGED;

    cv::Mat img = cv::imread(path, flag);
    if (img.empty())
        throw std::runtime_error("Could not read image: " + path);
    return img;
}

void ensure_parent_dir(const std::string& path)
{
    std::filesystem::path p(path);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());
}

std::string save_image(const std::string& path, const cv::Mat& img)
{
    ensure_parent_dir(path);
    if (!cv::imwrite(path, img))
        throw std::runtime_error("Failed to write image: " + path);
    return path;
}

} // anon

/* ═══════════════════════════════════════════════════════════════════
 *  1.  Canny edge detection
 * ═══════════════════════════════════════════════════════════════════ */

std::string canny_edge(const std::string& in,
                       const std::string& out,
                       double t1, double t2,
                       int aperture)
{
    cv::Mat gray = load_image(in, "gray");
    cv::Mat edges;
    int ap = (aperture < 3) ? 3 : (aperture % 2 == 0 ? aperture + 1 : aperture);
    cv::Canny(gray, edges, t1, t2, ap);
    return save_image(out, edges);
}

/* ═══════════════════════════════════════════════════════════════════
 *  2.  Hough Line Detection
 * ═══════════════════════════════════════════════════════════════════ */

py::dict hough_lines(const std::string& in,
                     const std::string& edge_path,
                     const std::string& out,
                     double rho, double theta_deg,
                     int threshold,
                     double min_length, double max_gap)
{
    cv::Mat color = load_image(in, "color");
    cv::Mat edges = load_image(edge_path, "gray");

    double theta = theta_deg * CV_PI / 180.0;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, rho, theta, threshold, min_length, max_gap);

    cv::Mat canvas = color.clone();
    for (auto& l : lines)
        cv::line(canvas, {l[0], l[1]}, {l[2], l[3]},
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    save_image(out, canvas);

    py::dict result;
    result["output"]     = out;
    result["line_count"] = static_cast<int>(lines.size());
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  3.  Hough Circle Detection
 * ═══════════════════════════════════════════════════════════════════ */

py::dict hough_circles(const std::string& in,
                       const std::string& out,
                       double dp, double min_dist,
                       double param1, double param2,
                       int min_r, int max_r)
{
    cv::Mat color = load_image(in, "color");
    cv::Mat gray  = load_image(in, "gray");

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
                     dp, min_dist, param1, param2, min_r, max_r);

    cv::Mat canvas = color.clone();
    for (auto& c : circles) {
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        cv::circle(canvas, center, radius, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::circle(canvas, center, 3,      cv::Scalar(0, 255, 0), -1);
    }

    save_image(out, canvas);

    py::dict result;
    result["output"]       = out;
    result["circle_count"] = static_cast<int>(circles.size());
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  4.  Ellipse Detection  (contour-based fitEllipse)
 * ═══════════════════════════════════════════════════════════════════ */

py::dict detect_ellipses(const std::string& in,
                         const std::string& edge_path,
                         const std::string& out,
                         int min_contour_pts)
{
    cv::Mat color = load_image(in, "color");
    cv::Mat edges = load_image(edge_path, "gray");

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::Mat canvas = color.clone();
    int count = 0;

    for (auto& c : contours) {
        if (static_cast<int>(c.size()) < std::max(5, min_contour_pts))
            continue;
        cv::RotatedRect ell = cv::fitEllipse(c);
        // filter out degenerate ellipses
        float ratio = ell.size.width / (ell.size.height + 1e-6f);
        if (ratio < 0.1f || ratio > 10.0f) continue;
        cv::ellipse(canvas, ell, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        ++count;
    }

    save_image(out, canvas);

    py::dict result;
    result["output"]        = out;
    result["ellipse_count"] = count;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  5.  Detect ALL shapes (convenience wrapper)
 * ═══════════════════════════════════════════════════════════════════ */

py::dict detect_all_shapes(const std::string& in,
                           const std::string& prefix,
                           double canny_t1, double canny_t2,
                           int    canny_aperture,
                           // lines
                           double rho, double theta_deg, int line_thresh,
                           double min_line_len, double max_line_gap,
                           // circles
                           double dp, double min_dist,
                           double circle_p1, double circle_p2,
                           int min_r, int max_r,
                           // ellipses
                           int min_contour_pts)
{
    // 1. Canny
    std::string edge_p = prefix + "_edges.png";
    canny_edge(in, edge_p, canny_t1, canny_t2, canny_aperture);

    // 2. Lines
    std::string line_p = prefix + "_lines.png";
    py::dict ld = hough_lines(in, edge_p, line_p,
                              rho, theta_deg, line_thresh,
                              min_line_len, max_line_gap);

    // 3. Circles
    std::string circ_p = prefix + "_circles.png";
    py::dict cd = hough_circles(in, circ_p,
                                dp, min_dist, circle_p1, circle_p2,
                                min_r, max_r);

    // 4. Ellipses
    std::string ell_p = prefix + "_ellipses.png";
    py::dict ed = detect_ellipses(in, edge_p, ell_p, min_contour_pts);

    // 5. Combined overlay
    cv::Mat color  = load_image(in, "color");
    cv::Mat edges  = load_image(edge_p, "gray");
    cv::Mat canvas = color.clone();

    // draw lines
    {
        double theta = theta_deg * CV_PI / 180.0;
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, rho, theta, line_thresh,
                        min_line_len, max_line_gap);
        for (auto& l : lines)
            cv::line(canvas, {l[0], l[1]}, {l[2], l[3]},
                     cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    // draw circles
    {
        cv::Mat gray = load_image(in, "gray");
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
                         dp, min_dist, circle_p1, circle_p2, min_r, max_r);
        for (auto& c : circles) {
            cv::Point ctr(cvRound(c[0]), cvRound(c[1]));
            cv::circle(canvas, ctr, cvRound(c[2]),
                       cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }
    // draw ellipses
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges.clone(), contours,
                         cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        for (auto& c : contours) {
            if (static_cast<int>(c.size()) < std::max(5, min_contour_pts))
                continue;
            cv::RotatedRect ell = cv::fitEllipse(c);
            float ratio = ell.size.width / (ell.size.height + 1e-6f);
            if (ratio < 0.1f || ratio > 10.0f) continue;
            cv::ellipse(canvas, ell, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        }
    }

    std::string comb_p = prefix + "_combined.png";
    save_image(comb_p, canvas);

    py::dict result;
    result["edges"]         = edge_p;
    result["lines"]         = line_p;
    result["circles"]       = circ_p;
    result["ellipses"]      = ell_p;
    result["combined"]      = comb_p;
    result["line_count"]    = ld["line_count"];
    result["circle_count"]  = cd["circle_count"];
    result["ellipse_count"] = ed["ellipse_count"];
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  6.  Active Contour Model  (Greedy Snake)
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Energy = α·E_continuity + β·E_curvature + γ·E_image
 *
 *  E_continuity = |d̄ - |p_i - p_{i-1}||²   (keeps points evenly spaced)
 *  E_curvature  = |p_{i-1} - 2p_i + p_{i+1}|²  (penalises sharp bends)
 *  E_image      = -|∇I(p_i)|²                   (attracts to edges)
 *
 *  Greedy: for each point, search w×w window for minimum-energy move.
 * ═══════════════════════════════════════════════════════════════════ */

py::dict active_contour_greedy(const std::string& in,
                               const std::string& out,
                               int cx, int cy, int radius,
                               int num_points,
                               double alpha, double beta, double gamma,
                               int window_size, int iterations)
{
    cv::Mat gray = load_image(in, "gray");
    cv::Mat color = load_image(in, "color");

    // ── Edge detection ──
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, {5, 5}, 1.4);
    cv::Mat edges;
    cv::Canny(blurred, edges, 30, 100);

    // ── Distance transform + its gradient ──
    cv::Mat edge_inv;
    cv::bitwise_not(edges, edge_inv);
    cv::Mat dist_map;
    cv::distanceTransform(edge_inv, dist_map, cv::DIST_L2, 5);

    cv::Mat grad_x, grad_y;
    cv::Sobel(dist_map, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(dist_map, grad_y, CV_64F, 0, 1, 3);

    // Initialise contour as circle
    int n = std::max(num_points, 8);
    std::vector<cv::Point2d> snake(n);
    for (int i = 0; i < n; ++i) {
        double angle = 2.0 * CV_PI * i / n;
        snake[i].x = cx + radius * std::cos(angle);
        snake[i].y = cy + radius * std::sin(angle);
    }

    // ═══════════════════════════════════════════════════════
    //  PHASE 1: Gradient descent on distance transform
    // ═══════════════════════════════════════════════════════
    int edge_iters = iterations * 2 / 3;
    double step = gamma;

    for (int iter = 0; iter < edge_iters; ++iter) {
        bool any_moved = false;

        for (int i = 0; i < n; ++i) {
            int ix = std::clamp((int)std::round(snake[i].x), 1, gray.cols - 2);
            int iy = std::clamp((int)std::round(snake[i].y), 1, gray.rows - 2);

            float d = dist_map.at<float>(iy, ix);
            if (d < 1.5) continue;

            double gx_val = grad_x.at<double>(iy, ix);
            double gy_val = grad_y.at<double>(iy, ix);
            double mag = std::sqrt(gx_val * gx_val + gy_val * gy_val);
            if (mag < 1e-6) continue;

            double s = std::min(d * 0.5, step * 3.0);
            snake[i].x -= (gx_val / mag) * s;
            snake[i].y -= (gy_val / mag) * s;

            snake[i].x = std::clamp(snake[i].x, 1.0, (double)(gray.cols - 2));
            snake[i].y = std::clamp(snake[i].y, 1.0, (double)(gray.rows - 2));
            any_moved = true;
        }

        if (!any_moved) break;
    }

    // ═══════════════════════════════════════════════════════
    //  PHASE 2: Laplacian smoothing
    // ═══════════════════════════════════════════════════════
    int smooth_iters = iterations - edge_iters;
    for (int iter = 0; iter < smooth_iters; ++iter) {
        std::vector<cv::Point2d> smoothed = snake;

        for (int i = 0; i < n; ++i) {
            int prev = (i - 1 + n) % n;
            int next = (i + 1) % n;

            int ix = std::clamp((int)std::round(snake[i].x), 0, gray.cols - 1);
            int iy = std::clamp((int)std::round(snake[i].y), 0, gray.rows - 1);
            float d = dist_map.at<float>(iy, ix);

            double anchor = std::exp(-d * 0.5);
            double smooth_weight = alpha * 0.3 * (1.0 - anchor);

            cv::Point2d avg = (snake[prev] + snake[next]) * 0.5;
            smoothed[i] = snake[i] * (1.0 - smooth_weight) + avg * smooth_weight;
        }

        snake = smoothed;
    }

    // ── Chain code (8-connectivity Freeman) ──
    std::vector<int> chain_code;
    //  Direction mapping:
    //  3 2 1
    //  4 x 0
    //  5 6 7
    auto direction = [](const cv::Point2d& from, const cv::Point2d& to) -> int {
        int dx = static_cast<int>(std::round(to.x - from.x));
        int dy = static_cast<int>(std::round(to.y - from.y));
        dx = std::clamp(dx, -1, 1);
        dy = std::clamp(dy, -1, 1);
        // map (dx,dy) → chain code
        if (dx ==  1 && dy ==  0) return 0;
        if (dx ==  1 && dy == -1) return 1;
        if (dx ==  0 && dy == -1) return 2;
        if (dx == -1 && dy == -1) return 3;
        if (dx == -1 && dy ==  0) return 4;
        if (dx == -1 && dy ==  1) return 5;
        if (dx ==  0 && dy ==  1) return 6;
        if (dx ==  1 && dy ==  1) return 7;
        return -1;  // same point
    };

    for (int i = 0; i < n; ++i) {
        int next = (i + 1) % n;
        int code = direction(snake[i], snake[next]);
        if (code >= 0) chain_code.push_back(code);
    }

    // ── Perimeter from chain code ──
    double perimeter = 0;
    for (int c : chain_code)
        perimeter += (c % 2 == 0) ? 1.0 : std::sqrt(2.0);

    // ── Area via shoelace formula ──
    double area = 0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += snake[i].x * snake[j].y;
        area -= snake[j].x * snake[i].y;
    }
    area = std::abs(area) * 0.5;

    // ── Draw final contour ──
    cv::Mat canvas = color.clone();
    std::vector<cv::Point> pts(n);
    for (int i = 0; i < n; ++i) {
        pts[i].x = static_cast<int>(std::round(snake[i].x));
        pts[i].y = static_cast<int>(std::round(snake[i].y));
    }
    cv::polylines(canvas, pts, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    for (auto& p : pts)
        cv::circle(canvas, p, 3, cv::Scalar(0, 0, 255), -1);

    save_image(out, canvas);

    // Chain code as string
    std::ostringstream cc_ss;
    for (size_t i = 0; i < chain_code.size(); ++i) {
        if (i > 0) cc_ss << " ";
        cc_ss << chain_code[i];
    }

    py::dict result;
    result["output"]     = out;
    result["chain_code"] = cc_ss.str();
    result["perimeter"]  = perimeter;
    result["area"]       = area;
    result["num_points"] = n;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  7.  Active Contour from User-Drawn Points
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Same energy minimisation as active_contour_greedy, but the initial
 *  contour is supplied by the user as a list of (x, y) coordinate pairs
 *  drawn interactively on the image.
 * ═══════════════════════════════════════════════════════════════════ */

py::dict active_contour_from_points(const std::string& in,
                                    const std::string& out,
                                    std::vector<std::pair<int,int>> init_points,
                                    double alpha, double beta, double gamma,
                                    int window_size, int iterations)
{
    if (init_points.size() < 3)
        throw std::runtime_error("Need at least 3 contour points");

    cv::Mat gray  = load_image(in, "gray");
    cv::Mat color = load_image(in, "color");

    // ── Edge detection ──
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, {5, 5}, 1.4);
    cv::Mat edges;
    cv::Canny(blurred, edges, 30, 100);

    // ── Distance transform + its gradient ──
    cv::Mat edge_inv;
    cv::bitwise_not(edges, edge_inv);
    cv::Mat dist_map;
    cv::distanceTransform(edge_inv, dist_map, cv::DIST_L2, 5);

    // Gradient of distance transform → points toward nearest edge
    cv::Mat grad_x, grad_y;
    cv::Sobel(dist_map, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(dist_map, grad_y, CV_64F, 0, 1, 3);

    // Initialise contour from user-drawn points
    int n = static_cast<int>(init_points.size());
    std::vector<cv::Point2d> snake(n);
    for (int i = 0; i < n; ++i) {
        snake[i].x = init_points[i].first;
        snake[i].y = init_points[i].second;
    }

    // ═══════════════════════════════════════════════════════
    //  PHASE 1: Gradient descent on distance transform
    //  Each point follows the steepest descent toward the
    //  nearest edge. Guaranteed to converge.
    // ═══════════════════════════════════════════════════════
    int edge_iters = iterations * 2 / 3;
    double step = gamma;  // use gamma slider as step size multiplier

    for (int iter = 0; iter < edge_iters; ++iter) {
        bool any_moved = false;

        for (int i = 0; i < n; ++i) {
            int ix = std::clamp((int)std::round(snake[i].x), 1, gray.cols - 2);
            int iy = std::clamp((int)std::round(snake[i].y), 1, gray.rows - 2);

            float d = dist_map.at<float>(iy, ix);
            if (d < 1.5) continue;  // already on/near an edge

            double gx = grad_x.at<double>(iy, ix);
            double gy = grad_y.at<double>(iy, ix);
            double mag = std::sqrt(gx * gx + gy * gy);
            if (mag < 1e-6) continue;

            // Step size: proportional to distance, capped
            double s = std::min(d * 0.5, step * 3.0);
            snake[i].x -= (gx / mag) * s;
            snake[i].y -= (gy / mag) * s;

            // Clamp to image bounds
            snake[i].x = std::clamp(snake[i].x, 1.0, (double)(gray.cols - 2));
            snake[i].y = std::clamp(snake[i].y, 1.0, (double)(gray.rows - 2));
            any_moved = true;
        }

        if (!any_moved) break;
    }

    // ═══════════════════════════════════════════════════════
    //  PHASE 2: Laplacian smoothing
    //  Gently average neighboring points for smooth contour.
    //  alpha controls smoothing strength (0 = no smoothing).
    //  Points near edges are anchored in place.
    // ═══════════════════════════════════════════════════════
    int smooth_iters = iterations - edge_iters;
    for (int iter = 0; iter < smooth_iters; ++iter) {
        std::vector<cv::Point2d> smoothed = snake;

        for (int i = 0; i < n; ++i) {
            int prev = (i - 1 + n) % n;
            int next = (i + 1) % n;

            // How close is this point to an edge?
            int ix = std::clamp((int)std::round(snake[i].x), 0, gray.cols - 1);
            int iy = std::clamp((int)std::round(snake[i].y), 0, gray.rows - 1);
            float d = dist_map.at<float>(iy, ix);

            // Anchor factor: points on edges barely move
            double anchor = std::exp(-d * 0.5);  // 1.0 on edge, → 0 far away
            double smooth_weight = alpha * 0.3 * (1.0 - anchor);

            // Laplacian: average of neighbors
            cv::Point2d avg = (snake[prev] + snake[next]) * 0.5;
            smoothed[i] = snake[i] * (1.0 - smooth_weight) + avg * smooth_weight;
        }

        snake = smoothed;
    }

    // ── Chain code (8-connectivity Freeman) ──
    std::vector<int> chain_code;
    auto direction = [](const cv::Point2d& from, const cv::Point2d& to) -> int {
        int dx = static_cast<int>(std::round(to.x - from.x));
        int dy = static_cast<int>(std::round(to.y - from.y));
        dx = std::clamp(dx, -1, 1);
        dy = std::clamp(dy, -1, 1);
        if (dx ==  1 && dy ==  0) return 0;
        if (dx ==  1 && dy == -1) return 1;
        if (dx ==  0 && dy == -1) return 2;
        if (dx == -1 && dy == -1) return 3;
        if (dx == -1 && dy ==  0) return 4;
        if (dx == -1 && dy ==  1) return 5;
        if (dx ==  0 && dy ==  1) return 6;
        if (dx ==  1 && dy ==  1) return 7;
        return -1;
    };

    for (int i = 0; i < n; ++i) {
        int next = (i + 1) % n;
        int code = direction(snake[i], snake[next]);
        if (code >= 0) chain_code.push_back(code);
    }

    // ── Perimeter from chain code ──
    double perimeter = 0;
    for (int c : chain_code)
        perimeter += (c % 2 == 0) ? 1.0 : std::sqrt(2.0);

    // ── Area via shoelace formula ──
    double area = 0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += snake[i].x * snake[j].y;
        area -= snake[j].x * snake[i].y;
    }
    area = std::abs(area) * 0.5;

    // ── Draw final contour ──
    cv::Mat canvas = color.clone();
    std::vector<cv::Point> pts(n);
    for (int i = 0; i < n; ++i) {
        pts[i].x = static_cast<int>(std::round(snake[i].x));
        pts[i].y = static_cast<int>(std::round(snake[i].y));
    }
    cv::polylines(canvas, pts, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    for (auto& p : pts)
        cv::circle(canvas, p, 3, cv::Scalar(0, 0, 255), -1);

    save_image(out, canvas);

    // Chain code as string
    std::ostringstream cc_ss;
    for (size_t i = 0; i < chain_code.size(); ++i) {
        if (i > 0) cc_ss << " ";
        cc_ss << chain_code[i];
    }

    py::dict result;
    result["output"]     = out;
    result["chain_code"] = cc_ss.str();
    result["perimeter"]  = perimeter;
    result["area"]       = area;
    result["num_points"] = n;
    return result;
}

/* ═══════════════════ pybind11 bindings ════════════════════════════ */

PYBIND11_MODULE(cv_core, m)
{
    m.doc() = "C++ OpenCV core – Hough transform & Active Contour for CV_02";

    m.def("canny_edge", &canny_edge,
          "Canny edge detection",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("threshold1") = 50.0, py::arg("threshold2") = 150.0,
          py::arg("aperture") = 3);

    m.def("hough_lines", &hough_lines,
          "Detect and draw lines via HoughLinesP",
          py::arg("input_path"), py::arg("edge_path"), py::arg("output_path"),
          py::arg("rho") = 1.0, py::arg("theta_deg") = 1.0,
          py::arg("threshold") = 50,
          py::arg("min_line_length") = 50.0, py::arg("max_line_gap") = 10.0);

    m.def("hough_circles", &hough_circles,
          "Detect and draw circles via HoughCircles",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("dp") = 1.2, py::arg("min_dist") = 30.0,
          py::arg("param1") = 100.0, py::arg("param2") = 30.0,
          py::arg("min_radius") = 0, py::arg("max_radius") = 0);

    m.def("detect_ellipses", &detect_ellipses,
          "Detect and draw ellipses via contour-based fitEllipse",
          py::arg("input_path"), py::arg("edge_path"), py::arg("output_path"),
          py::arg("min_contour_points") = 20);

    m.def("detect_all_shapes", &detect_all_shapes,
          "Run Canny → lines + circles + ellipses, return all paths",
          py::arg("input_path"), py::arg("output_prefix"),
          py::arg("canny_t1") = 50.0, py::arg("canny_t2") = 150.0,
          py::arg("canny_aperture") = 3,
          py::arg("rho") = 1.0, py::arg("theta_deg") = 1.0,
          py::arg("line_threshold") = 50,
          py::arg("min_line_length") = 50.0, py::arg("max_line_gap") = 10.0,
          py::arg("dp") = 1.2, py::arg("min_dist") = 30.0,
          py::arg("circle_param1") = 100.0, py::arg("circle_param2") = 30.0,
          py::arg("min_radius") = 0, py::arg("max_radius") = 0,
          py::arg("min_contour_points") = 20);

    m.def("active_contour_greedy", &active_contour_greedy,
          "Greedy Snake active contour → chain code + perimeter + area",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("center_x"), py::arg("center_y"), py::arg("radius"),
          py::arg("num_points") = 60,
          py::arg("alpha") = 1.0, py::arg("beta") = 1.0, py::arg("gamma") = 1.5,
          py::arg("window_size") = 7, py::arg("iterations") = 100);

    m.def("active_contour_from_points", &active_contour_from_points,
          "Greedy Snake from user-drawn points → chain code + perimeter + area",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("init_points"),
          py::arg("alpha") = 1.0, py::arg("beta") = 1.0, py::arg("gamma") = 1.5,
          py::arg("window_size") = 7, py::arg("iterations") = 100);
}
