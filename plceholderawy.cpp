/**
 * cv_core – C++ OpenCV operations exposed to Python via pybind11.
 *
 * Every computer-vision operation lives here; the Python / Django layer
 * is only a thin HTTP wrapper and must NEVER call cv2 directly.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

/* ───────────────────── internal helpers ───────────────────── */
namespace {

cv::Mat load_image(const std::string& path, const std::string& mode)
{
    int flag = cv::IMREAD_COLOR;
    if (mode == "gray")
        flag = cv::IMREAD_GRAYSCALE;
    else if (mode == "unchanged")
        flag = cv::IMREAD_UNCHANGED;

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

int ensure_odd_min3(int v)
{
    if (v < 3) v = 3;
    if (v % 2 == 0) v += 1;
    return v;
}

cv::Mat abs_preview(const cv::Mat& src)
{
    cv::Mat out;
    cv::convertScaleAbs(src, out);
    return out;
}

cv::Mat magnitude_preview(const cv::Mat& gx, const cv::Mat& gy)
{
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    cv::Mat out;
    mag.convertTo(out, CV_8U);
    return out;
}

/*
 * dft_shift
 * ─────────
 * Shifts the zero-frequency component to the centre of the spectrum.
 * NOTE: This operates IN-PLACE and also crops the matrix to even dimensions.
 *       The returned size may differ from the input; callers that need the
 *       exact post-crop size should query m.size() afterwards.
 */
void dft_shift(cv::Mat& m)
{
    // Crop to even dimensions
    m = m(cv::Rect(0, 0, m.cols & -2, m.rows & -2));
    int cx = m.cols / 2, cy = m.rows / 2;

    cv::Mat q0(m, cv::Rect(0,  0,  cx, cy));
    cv::Mat q1(m, cv::Rect(cx, 0,  cx, cy));
    cv::Mat q2(m, cv::Rect(0,  cy, cx, cy));
    cv::Mat q3(m, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

/*
 * draw_channel_hist_and_cdf
 * ─────────────────────────
 * Draws the histogram (left half) and CDF (right half) for a single
 * grayscale channel onto a 1024x400 canvas and returns it.
 * color – BGR scalar for the drawn lines.
 */
cv::Mat draw_channel_hist_and_cdf(const cv::Mat& channel,
                                  const cv::Scalar& color)
{
    constexpr int BINS = 256;
    constexpr int PW = 512, PH = 400, W = PW * 2, H = PH;
    const int bin_w = cvRound(static_cast<double>(PW) / BINS);

    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    int   ch[]    = {0};
    int   bins[]  = {BINS};
    float range[] = {0, 256};
    const float* ranges[] = {range};

    cv::Mat hist;
    cv::calcHist(&channel, 1, ch, {}, hist, 1, bins, ranges);

    cv::Mat hn;
    cv::normalize(hist, hn, 0, PH, cv::NORM_MINMAX);

    // Build CDF as cumulative sum of histogram
    cv::Mat cdf = hist.clone();
    for (int i = 1; i < cdf.rows; ++i)
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    cv::Mat cn;
    cv::normalize(cdf, cn, 0, PH, cv::NORM_MINMAX);

    // Left panel: histogram
    for (int i = 1; i < BINS; ++i) {
        cv::line(canvas,
                 {bin_w*(i-1),      PH - cvRound(hn.at<float>(i-1))},
                 {bin_w*i,          PH - cvRound(hn.at<float>(i))},
                 color, 2);
    }

    // Right panel: CDF (offset by PW)
    for (int i = 1; i < BINS; ++i) {
        cv::line(canvas,
                 {PW + bin_w*(i-1), PH - cvRound(cn.at<float>(i-1))},
                 {PW + bin_w*i,     PH - cvRound(cn.at<float>(i))},
                 color, 2);
    }

    // Labels
    cv::putText(canvas, "Histogram",
                {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(80,80,80), 1, cv::LINE_AA);
    cv::putText(canvas, "CDF (mapping function)",
                {PW + 10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(80,80,80), 1, cv::LINE_AA);

    // Divider
    cv::line(canvas, {PW, 0}, {PW, H}, cv::Scalar(180,180,180), 1);

    return canvas;
}

} // anonymous namespace

/* ───────────────── 1. Read image info ─────────────────────── */

py::dict read_image_info(const std::string& path)
{
    cv::Mat color = load_image(path, "color");
    cv::Mat gray  = load_image(path, "gray");

    py::dict info;
    info["input_path"]    = path;
    info["width"]         = color.cols;
    info["height"]        = color.rows;
    info["channels"]      = color.channels();
    info["gray_channels"] = gray.channels();
    return info;
}

/* ───────────────── 2. Convert to grayscale ────────────────── */

std::string save_grayscale(const std::string& in, const std::string& out)
{
    cv::Mat img = load_image(in, "color");
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return save_image(out, gray);
}

/* ───────────────── 3. Add noise ───────────────────────────── */

std::string add_noise(const std::string& in,
                      const std::string& out,
                      const std::string& type,
                      double amount,
                      double sigma,
                      int uniform_range)
{
    cv::Mat img = load_image(in, "color");
    cv::Mat result = img.clone();

    if (type == "gaussian") {
        cv::Mat flt, noise;
        img.convertTo(flt, CV_32F);
        noise = cv::Mat::zeros(img.size(), flt.type());
        cv::randn(noise, 0.0, sigma);
        cv::Mat noisy = flt + noise;
        noisy.convertTo(result, img.type());

    } else if (type == "uniform") {
        cv::Mat flt, noise;
        img.convertTo(flt, CV_32F);
        noise = cv::Mat::zeros(img.size(), flt.type());
        cv::randu(noise, -uniform_range, uniform_range);
        cv::Mat noisy = flt + noise;
        noisy.convertTo(result, img.type());

    } else if (type == "salt_pepper") {
        amount = std::clamp(amount, 0.0, 1.0);
        int total = img.rows * img.cols;
        int count = static_cast<int>(total * amount * 0.5);
        cv::RNG rng(static_cast<uint64>(cv::getTickCount()));

        for (int i = 0; i < count; ++i) {
            int y = rng.uniform(0, img.rows), x = rng.uniform(0, img.cols);
            if (result.channels() == 1) result.at<uchar>(y, x) = 255;
            else result.at<cv::Vec3b>(y, x) = {255, 255, 255};
        }
        for (int i = 0; i < count; ++i) {
            int y = rng.uniform(0, img.rows), x = rng.uniform(0, img.cols);
            if (result.channels() == 1) result.at<uchar>(y, x) = 0;
            else result.at<cv::Vec3b>(y, x) = {0, 0, 0};
        }
    } else {
        throw std::runtime_error("Unknown noise type: " + type);
    }

    return save_image(out, result);
}

/* ───────────────── 4. Low-pass filter ─────────────────────── */

std::string apply_low_pass_filter(const std::string& in,
                                  const std::string& out,
                                  const std::string& type,
                                  int ksize,
                                  double sigma)
{
    cv::Mat img = load_image(in, "color");
    cv::Mat result;
    ksize = ensure_odd_min3(ksize);

    if (type == "average") {
        cv::blur(img, result, {ksize, ksize});
    } else if (type == "gaussian") {
        cv::GaussianBlur(img, result, {ksize, ksize}, sigma, sigma);
    } else if (type == "median") {
        cv::medianBlur(img, result, ksize);
    } else {
        throw std::runtime_error("Unknown filter type: " + type);
    }

    return save_image(out, result);
}

/* ───────────────── 5. Edge detection ──────────────────────── */

py::dict detect_edges(const std::string& in,
                      const std::string& prefix,
                      const std::string& method,
                      int ksize,
                      double t1,
                      double t2)
{
    cv::Mat gray = load_image(in, "gray");
    py::dict out;
    out["method"] = method;

    if (method == "canny") {
        cv::Mat edges;
        cv::Canny(gray, edges, t1, t2);
        std::string p = prefix + "_canny.png";
        save_image(p, edges);
        out["edge"] = p;
        return out;
    }

    cv::Mat gx, gy;
    if (method == "sobel") {
        ksize = ensure_odd_min3(ksize);
        cv::Sobel(gray, gx, CV_32F, 1, 0, ksize);
        cv::Sobel(gray, gy, CV_32F, 0, 1, ksize);
    } else if (method == "prewitt") {
        cv::Mat kx = (cv::Mat_<float>(3,3) << -1,0,1, -1,0,1, -1,0,1);
        cv::Mat ky = (cv::Mat_<float>(3,3) << -1,-1,-1, 0,0,0, 1,1,1);
        cv::filter2D(gray, gx, CV_32F, kx);
        cv::filter2D(gray, gy, CV_32F, ky);
    } else if (method == "roberts") {
        cv::Mat kx = (cv::Mat_<float>(2,2) << 1,0, 0,-1);
        cv::Mat ky = (cv::Mat_<float>(2,2) << 0,1, -1,0);
        cv::filter2D(gray, gx, CV_32F, kx);
        cv::filter2D(gray, gy, CV_32F, ky);
    } else {
        throw std::runtime_error("Unknown edge method: " + method);
    }

    std::string xp = prefix + "_" + method + "_x.png";
    std::string yp = prefix + "_" + method + "_y.png";
    std::string mp = prefix + "_" + method + "_edge.png";

    save_image(xp, abs_preview(gx));
    save_image(yp, abs_preview(gy));
    save_image(mp, magnitude_preview(gx, gy));

    out["x"]    = xp;
    out["y"]    = yp;
    out["edge"] = mp;
    return out;
}

/* ───────────────── 6. Histogram & CDF ─────────────────────── */

py::dict draw_histogram_and_cdf(const std::string& in,
                                const std::string& hist_path,
                                const std::string& cdf_path,
                                const std::string& mode)
{
    cv::Mat img = load_image(in, mode == "rgb" ? "color" : "gray");

    constexpr int BINS = 256, W = 512, H = 400;
    const int bin_w = cvRound(static_cast<double>(W) / BINS);

    cv::Mat hist_canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat cdf_canvas (H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    std::vector<cv::Scalar> colors = {{255,0,0}, {0,255,0}, {0,0,255}};
    std::vector<cv::Mat> channels;

    if (img.channels() == 1) {
        channels.push_back(img);
        colors = {{0, 0, 0}};
    } else {
        cv::split(img, channels);
    }

    for (size_t c = 0; c < channels.size(); ++c) {
        int ch[]  = {0};
        int bins[] = {BINS};
        float range[] = {0, 256};
        const float* ranges[] = {range};

        cv::Mat hist;
        cv::calcHist(&channels[c], 1, ch, {}, hist, 1, bins, ranges);

        cv::Mat hn;
        cv::normalize(hist, hn, 0, H, cv::NORM_MINMAX);

        cv::Mat cdf = hist.clone();
        for (int i = 1; i < cdf.rows; ++i)
            cdf.at<float>(i) += cdf.at<float>(i - 1);
        cv::Mat cn;
        cv::normalize(cdf, cn, 0, H, cv::NORM_MINMAX);

        for (int i = 1; i < BINS; ++i) {
            cv::line(hist_canvas,
                     {bin_w*(i-1), H - cvRound(hn.at<float>(i-1))},
                     {bin_w*i,     H - cvRound(hn.at<float>(i))},
                     colors[c], 2);
            cv::line(cdf_canvas,
                     {bin_w*(i-1), H - cvRound(cn.at<float>(i-1))},
                     {bin_w*i,     H - cvRound(cn.at<float>(i))},
                     colors[c], 2);
        }
    }

    save_image(hist_path, hist_canvas);
    save_image(cdf_path,  cdf_canvas);

    py::dict out;
    out["histogram"] = hist_path;
    out["cdf"]       = cdf_path;
    out["mode"]      = mode;
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  7. Histogram equalization
 * ═══════════════════════════════════════════════════════════════ */

py::dict equalize_image(const std::string& in,
                        const std::string& prefix)
{
    cv::Mat color = load_image(in, "color");     // BGR

    // Grayscale: convert then equalize
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray_eq;
    cv::equalizeHist(gray, gray_eq);

    // Color: equalize each channel independently
    std::vector<cv::Mat> channels(3);
    cv::split(color, channels);                  // [0]=B [1]=G [2]=R

    std::vector<cv::Mat> eq_ch(3);
    for (int c = 0; c < 3; ++c)
        cv::equalizeHist(channels[c], eq_ch[c]);

    cv::Mat color_eq;
    cv::merge(eq_ch, color_eq);

    // Save output images
    std::string p_gray     = prefix + "_eq_gray.png";
    std::string p_color_eq = prefix + "_eq_color.png";
    save_image(p_gray,     gray_eq);
    save_image(p_color_eq, color_eq);

    // Color scalars (BGR order)
    cv::Scalar colK(  0,   0,   0);
    cv::Scalar colB(200,  80,  80);
    cv::Scalar colG( 50, 160,  50);
    cv::Scalar colR( 50,  50, 200);

    // BEFORE graphs
    std::string p_bef_gray = prefix + "_bef_gray.png";
    std::string p_bef_b    = prefix + "_bef_b.png";
    std::string p_bef_g    = prefix + "_bef_g.png";
    std::string p_bef_r    = prefix + "_bef_r.png";

    save_image(p_bef_gray, draw_channel_hist_and_cdf(gray,        colK));
    save_image(p_bef_b,    draw_channel_hist_and_cdf(channels[0], colB));
    save_image(p_bef_g,    draw_channel_hist_and_cdf(channels[1], colG));
    save_image(p_bef_r,    draw_channel_hist_and_cdf(channels[2], colR));

    // AFTER graphs
    std::string p_aft_gray = prefix + "_aft_gray.png";
    std::string p_aft_b    = prefix + "_aft_b.png";
    std::string p_aft_g    = prefix + "_aft_g.png";
    std::string p_aft_r    = prefix + "_aft_r.png";

    save_image(p_aft_gray, draw_channel_hist_and_cdf(gray_eq,  colK));
    save_image(p_aft_b,    draw_channel_hist_and_cdf(eq_ch[0], colB));
    save_image(p_aft_g,    draw_channel_hist_and_cdf(eq_ch[1], colG));
    save_image(p_aft_r,    draw_channel_hist_and_cdf(eq_ch[2], colR));

    py::dict result;
    result["output_gray"]     = p_gray;
    result["output_color_eq"] = p_color_eq;
    result["before_gray"]     = p_bef_gray;
    result["before_b"]        = p_bef_b;
    result["before_g"]        = p_bef_g;
    result["before_r"]        = p_bef_r;
    result["after_gray"]      = p_aft_gray;
    result["after_b"]         = p_aft_b;
    result["after_g"]         = p_aft_g;
    result["after_r"]         = p_aft_r;
    return result;
}

/* ───────────────── 8. Normalize ───────────────────────────── */

std::string normalize_image(const std::string& in,
                            const std::string& out,
                            double alpha,
                            double beta,
                            const std::string& norm_type)
{
    cv::Mat img = load_image(in, "color");

    int cv_norm = cv::NORM_MINMAX;
    if (norm_type == "inf") cv_norm = cv::NORM_INF;

    cv::Mat dst;
    cv::normalize(img, dst, alpha, beta, cv_norm, img.type());
    return save_image(out, dst);
}

/* ───────────────────────────────────────────────────────────────
 *  9. Thresholding
 * ─────────────────────────────────────────────────────────────── */

std::string apply_threshold(const std::string& in,
                            const std::string& out,
                            double thresh,
                            double max_val)
{
    cv::Mat gray = load_image(in, "gray");
    cv::Mat dst;
    cv::threshold(gray, dst, thresh, max_val, cv::THRESH_BINARY);
    return save_image(out, dst);
}

/* ═══════════════════════════════════════════════════════════════
 *  10. Color → Grayscale Transformation
 * ═══════════════════════════════════════════════════════════════ */

py::dict color_to_gray_transform(const std::string& in,
                                 const std::string& prefix)
{
    cv::Mat color = load_image(in, "color");

    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    std::string p_gray = prefix + "_transform_gray.png";
    save_image(p_gray, gray);

    std::vector<cv::Mat> channels(3);
    cv::split(color, channels);

    cv::Scalar colK(  0,   0,   0);
    cv::Scalar colB(200,  80,  80);
    cv::Scalar colG( 50, 160,  50);
    cv::Scalar colR( 50,  50, 200);

    std::string p_hist_gray = prefix + "_transform_hist_gray.png";
    std::string p_hist_b    = prefix + "_transform_hist_b.png";
    std::string p_hist_g    = prefix + "_transform_hist_g.png";
    std::string p_hist_r    = prefix + "_transform_hist_r.png";

    save_image(p_hist_gray, draw_channel_hist_and_cdf(gray,        colK));
    save_image(p_hist_b,    draw_channel_hist_and_cdf(channels[0], colB));
    save_image(p_hist_g,    draw_channel_hist_and_cdf(channels[1], colG));
    save_image(p_hist_r,    draw_channel_hist_and_cdf(channels[2], colR));

    py::dict result;
    result["output_gray"] = p_gray;
    result["hist_gray"]   = p_hist_gray;
    result["hist_b"]      = p_hist_b;
    result["hist_g"]      = p_hist_g;
    result["hist_r"]      = p_hist_r;
    return result;
}

/* ═══════════════════════════════════════════════════════════════
 *  11a. Frequency-domain filter  (FIX: mask built from DFT size)
 * ═══════════════════════════════════════════════════════════════
 *
 * ROOT CAUSE of the original crash
 * ──────────────────────────────────
 * cv::mulSpectrums requires both operands to have identical size AND type.
 * The old code:
 *   1. Created `cpx` (complex) from the original gray image size.
 *   2. Called dft_shift(cpx) which crops cpx to even dimensions – so
 *      cpx.size() may now differ from gray.size().
 *   3. Built `mask` from gray.size() – WRONG, it must match cpx.size().
 *   4. mulSpectrums(cpx, cmask, …) → size mismatch → assertion failure.
 *
 * FIX: build the mask AFTER the DFT+shift so we use cpx.size().
 *      Also ensure the mask is CV_32F (same type as the DFT planes)
 *      before merging it into the 2-channel complex mask.
 * ═══════════════════════════════════════════════════════════════ */

std::string frequency_filter(const std::string& in,
                             const std::string& out,
                             const std::string& type,
             7                int cutoff)
{
    cv::Mat gray = load_image(in, "gray");

    // Pad to optimal DFT size for speed
    int optRows = cv::getOptimalDFTSize(gray.rows);
    int optCols = cv::getOptimalDFTSize(gray.cols);
    cv::Mat padded;
    cv::copyMakeBorder(gray, padded,
                       0, optRows - gray.rows,
                       0, optCols - gray.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Build complex matrix and compute DFT
    cv::Mat flt;
    padded.convertTo(flt, CV_32F);
    cv::Mat planes[] = {flt, cv::Mat::zeros(flt.size(), CV_32F)};
    cv::Mat cpx;
    cv::merge(planes, 2, cpx);
    cv::dft(cpx, cpx);

    // Shift zero-frequency to centre; cpx may be cropped to even dims
    dft_shift(cpx);

    // ── FIX: build mask from cpx.size() (post-crop), not gray.size() ──
    cutoff = std::max(1, cutoff);
    cv::Mat mask(cpx.rows, cpx.cols, CV_32F, cv::Scalar(0.0f));
    cv::circle(mask,
               cv::Point(mask.cols / 2, mask.rows / 2),
               cutoff,
               cv::Scalar(1.0f),
               -1);

    if (type == "high_pass")
        mask = cv::Scalar(1.0f) - mask;

    // Merge single-channel mask into a 2-channel complex mask
    cv::Mat fp[] = {mask, mask};
    cv::Mat cmask;
    cv::merge(fp, 2, cmask);   // same size & type as cpx ✓

    // Apply mask in frequency domain
    cv::mulSpectrums(cpx, cmask, cpx, 0);

    // Inverse shift + IDFT
    dft_shift(cpx);
    cv::Mat inv;
    cv::idft(cpx, inv, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    // Crop back to original image size and normalise
    cv::Mat cropped = inv(cv::Rect(0, 0,
                                   std::min(inv.cols, gray.cols),
                                   std::min(inv.rows, gray.rows)));
    cv::Mat norm;
    cv::normalize(cropped, norm, 0, 255, cv::NORM_MINMAX);

    cv::Mat dst;
    norm.convertTo(dst, CV_8U);
    return save_image(out, dst);
}

/* ───────────────── 11b. Hybrid image ──────────────────────── */

std::string hybrid_image(const std::string& low_path,
                         const std::string& high_path,
                         const std::string& out,
                         double low_sigma,
                         double high_sigma,
                         double mix)
{
    cv::Mat lo = load_image(low_path,  "color");
    cv::Mat hi = load_image(high_path, "color");

    if (lo.size() != hi.size())
        cv::resize(hi, hi, lo.size());

    cv::Mat lo_blur, hi_blur;
    cv::GaussianBlur(lo, lo_blur, {0, 0}, low_sigma);
    cv::GaussianBlur(hi, hi_blur, {0, 0}, high_sigma);

    cv::Mat hi_pass;
    cv::subtract(hi, hi_blur, hi_pass, cv::noArray(), CV_16S);

    cv::Mat lf, hf;
    lo_blur.convertTo(lf, CV_32F);
    hi_pass.convertTo(hf, CV_32F);

    mix = std::clamp(mix, 0.0, 1.0);
    cv::Mat hybrid = lf * mix + hf * (1.0 - mix);
    cv::normalize(hybrid, hybrid, 0, 255, cv::NORM_MINMAX);

    cv::Mat dst;
    hybrid.convertTo(dst, CV_8U);
    return save_image(out, dst);
}

/* ═══════════════════ pybind11 bindings ════════════════════════ */

PYBIND11_MODULE(cv_core, m)
{
    m.doc() = "C++ OpenCV core – all CV operations for CV_One";

    m.def("read_image_info", &read_image_info,
          "Read RGB / gray image metadata",
          py::arg("input_path"));

    m.def("save_grayscale", &save_grayscale,
          "Save grayscale copy of image",
          py::arg("input_path"), py::arg("output_path"));

    m.def("add_noise", &add_noise,
          "Add noise: gaussian | uniform | salt_pepper",
          py::arg("input_path"), py::arg("output_path"), py::arg("noise_type"),
          py::arg("amount") = 0.05, py::arg("sigma") = 25.0,
          py::arg("uniform_range") = 30);

    m.def("apply_low_pass_filter", &apply_low_pass_filter,
          "Low-pass filter: average | gaussian | median",
          py::arg("input_path"), py::arg("output_path"), py::arg("filter_type"),
          py::arg("kernel_size") = 3, py::arg("sigma") = 1.0);

    m.def("detect_edges", &detect_edges,
          "Edge detection: sobel | prewitt | roberts | canny",
          py::arg("input_path"), py::arg("output_prefix"), py::arg("method"),
          py::arg("kernel_size") = 3,
          py::arg("threshold1") = 50.0, py::arg("threshold2") = 150.0);

    m.def("draw_histogram_and_cdf", &draw_histogram_and_cdf,
          "Draw histogram and CDF curves",
          py::arg("input_path"), py::arg("hist_output_path"),
          py::arg("cdf_output_path"), py::arg("mode") = "gray");

    m.def("equalize_image", &equalize_image,
          "Histogram equalization with full before/after hist+CDF for gray+RGB",
          py::arg("input_path"), py::arg("output_prefix"));

    m.def("normalize_image", &normalize_image,
          "Normalize: minmax | inf",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("alpha") = 0.0, py::arg("beta") = 255.0,
          py::arg("norm_type") = "minmax");

    m.def("apply_threshold", &apply_threshold,
          "Binary thresholding (THRESH_BINARY: pixel > thresh → max_val, else 0)",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("threshold") = 127.0, py::arg("max_value") = 255.0);

    m.def("frequency_filter", &frequency_filter,
          "Frequency-domain filter: low_pass | high_pass",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("filter_type") = "low_pass", py::arg("cutoff") = 30);

    m.def("hybrid_image", &hybrid_image,
          "Hybrid image from low + high frequency sources",
          py::arg("low_image_path"), py::arg("high_image_path"),
          py::arg("output_path"),
          py::arg("low_sigma") = 5.0, py::arg("high_sigma") = 3.0,
          py::arg("mix_weight") = 0.5);

    m.def("color_to_gray_transform", &color_to_gray_transform,
          "Convert color to grayscale; plot R,G,B,Gray histograms + CDFs",
          py::arg("input_path"), py::arg("output_prefix"));
}