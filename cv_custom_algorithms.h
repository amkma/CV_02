#pragma once
/**
 * cv_custom_algorithms.h - From-scratch CV algorithms replacing OpenCV built-ins.
 * Only cv::Mat and basic OpenCV types are used for storage.
 */

#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <numeric>

namespace custom {

constexpr double PI = 3.14159265358979323846;

/* ─── pixel access helpers ─────────────────────────────────────── */

inline double get_pix(const cv::Mat& m, int y, int x) {
    if (m.type() == CV_8UC1)  return m.at<uchar>(y, x);
    if (m.type() == CV_32FC1) return m.at<float>(y, x);
    if (m.type() == CV_64FC1) return m.at<double>(y, x);
    return 0;
}

inline void set_pixel_color(cv::Mat& img, int x, int y, const cv::Scalar& c) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
    if (img.channels() == 3)
        img.at<cv::Vec3b>(y, x) = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
    else
        img.at<uchar>(y, x) = (uchar)c[0];
}

/* ─── Gaussian Blur ────────────────────────────────────────────── */

inline cv::Mat gaussian_blur(const cv::Mat& src, int ksize, double sigma) {
    int half = ksize / 2;
    std::vector<double> k(ksize);
    double sum = 0;
    for (int i = 0; i < ksize; ++i) {
        double x = i - half;
        k[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;

    // horizontal pass
    cv::Mat tmp(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            double v = 0;
            for (int i = -half; i <= half; ++i)
                v += get_pix(src, y, std::clamp(x + i, 0, src.cols - 1)) * k[i + half];
            tmp.at<double>(y, x) = v;
        }
    // vertical pass
    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            double v = 0;
            for (int i = -half; i <= half; ++i)
                v += tmp.at<double>(std::clamp(y + i, 0, src.rows - 1), x) * k[i + half];
            dst.at<uchar>(y, x) = (uchar)std::clamp(v, 0.0, 255.0);
        }
    return dst;
}

/* ─── Sobel operator (3×3, output CV_64F) ──────────────────────── */

inline cv::Mat sobel(const cv::Mat& src, int dx, int dy) {
    // Gx kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
    // Gy kernel: [[-1,-2,-1],[0,0,0],[1,2,1]]
    double Kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    double Ky[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    cv::Mat dst(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 1; y < src.rows - 1; ++y)
        for (int x = 1; x < src.cols - 1; ++x) {
            double val = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    double p = get_pix(src, y + ky, x + kx);
                    if (dx == 1) val += p * Kx[ky + 1][kx + 1];
                    else         val += p * Ky[ky + 1][kx + 1];
                }
            dst.at<double>(y, x) = val;
        }
    return dst;
}

/* ─── Canny Edge Detection ─────────────────────────────────────── */

inline cv::Mat canny(const cv::Mat& gray, double low_t, double high_t, int aperture) {
    int rows = gray.rows, cols = gray.cols;

    // 1. Gaussian blur
    double sigma = 0.3 * ((aperture - 1) * 0.5 - 1) + 0.8;
    if (sigma < 0.5) sigma = 0.5;
    cv::Mat blurred = gaussian_blur(gray, aperture < 5 ? 5 : aperture, sigma);

    // 2. Sobel gradients
    cv::Mat gx = sobel(blurred, 1, 0);
    cv::Mat gy = sobel(blurred, 0, 1);

    // 3. Magnitude & direction
    cv::Mat mag(rows, cols, CV_64FC1), dir(rows, cols, CV_64FC1);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            double vx = gx.at<double>(y, x), vy = gy.at<double>(y, x);
            mag.at<double>(y, x) = std::sqrt(vx * vx + vy * vy);
            dir.at<double>(y, x) = std::atan2(vy, vx);
        }

    // 4. Non-maximum suppression
    cv::Mat nms(rows, cols, CV_64FC1, cv::Scalar(0));
    for (int y = 1; y < rows - 1; ++y)
        for (int x = 1; x < cols - 1; ++x) {
            double angle = dir.at<double>(y, x) * 180.0 / PI;
            if (angle < 0) angle += 180.0;
            double m = mag.at<double>(y, x);
            double n1 = 0, n2 = 0;
            if ((angle < 22.5) || (angle >= 157.5))
                { n1 = mag.at<double>(y, x + 1); n2 = mag.at<double>(y, x - 1); }
            else if (angle < 67.5)
                { n1 = mag.at<double>(y - 1, x + 1); n2 = mag.at<double>(y + 1, x - 1); }
            else if (angle < 112.5)
                { n1 = mag.at<double>(y - 1, x); n2 = mag.at<double>(y + 1, x); }
            else
                { n1 = mag.at<double>(y - 1, x - 1); n2 = mag.at<double>(y + 1, x + 1); }
            if (m >= n1 && m >= n2) nms.at<double>(y, x) = m;
        }

    // 5. Hysteresis thresholding
    cv::Mat edges(rows, cols, CV_8UC1, cv::Scalar(0));
    // mark strong and weak
    enum : uchar { STRONG = 255, WEAK = 128 };
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            double v = nms.at<double>(y, x);
            if (v >= high_t) edges.at<uchar>(y, x) = STRONG;
            else if (v >= low_t) edges.at<uchar>(y, x) = WEAK;
        }
    // BFS: promote weak pixels connected to strong
    std::queue<cv::Point> q;
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            if (edges.at<uchar>(y, x) == STRONG) q.push({x, y});
    while (!q.empty()) {
        auto p = q.front(); q.pop();
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int ny = p.y + dy, nx = p.x + dx;
                if (ny >= 0 && ny < rows && nx >= 0 && nx < cols &&
                    edges.at<uchar>(ny, nx) == WEAK) {
                    edges.at<uchar>(ny, nx) = STRONG;
                    q.push({nx, ny});
                }
            }
    }
    // remove remaining weak
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            if (edges.at<uchar>(y, x) != STRONG) edges.at<uchar>(y, x) = 0;
    return edges;
}

/* ─── Hough Lines P ────────────────────────────────────────────── */

inline std::vector<cv::Vec4i> hough_lines_p(const cv::Mat& edges, double rho_res,
    double theta_res, int threshold, double minLen, double maxGap)
{
    int rows = edges.rows, cols = edges.cols;
    int nTheta = (int)(PI / theta_res);
    double diag = std::sqrt((double)(rows * rows + cols * cols));
    int nRho = (int)(2.0 * diag / rho_res + 1);
    double rhoOff = diag;

    std::vector<double> cosT(nTheta), sinT(nTheta);
    for (int t = 0; t < nTheta; ++t) {
        double th = t * theta_res;
        cosT[t] = std::cos(th); sinT[t] = std::sin(th);
    }

    // accumulator
    std::vector<int> acc(nRho * nTheta, 0);
    std::vector<cv::Point> pts;
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            if (edges.at<uchar>(y, x)) pts.push_back({x, y});

    for (auto& p : pts)
        for (int t = 0; t < nTheta; ++t) {
            int ri = (int)((p.x * cosT[t] + p.y * sinT[t] + rhoOff) / rho_res + 0.5);
            if (ri >= 0 && ri < nRho) acc[ri * nTheta + t]++;
        }

    // peaks
    struct Peak { int ri, ti, v; };
    std::vector<Peak> peaks;
    for (int ri = 0; ri < nRho; ++ri)
        for (int ti = 0; ti < nTheta; ++ti) {
            int v = acc[ri * nTheta + ti];
            if (v < threshold) continue;
            bool mx = true;
            for (int dr = -2; dr <= 2 && mx; ++dr)
                for (int dt = -2; dt <= 2 && mx; ++dt) {
                    if (!dr && !dt) continue;
                    int r2 = ri + dr, t2 = ti + dt;
                    if (r2 >= 0 && r2 < nRho && t2 >= 0 && t2 < nTheta)
                        if (acc[r2 * nTheta + t2] > v) mx = false;
                }
            if (mx) peaks.push_back({ri, ti, v});
        }
    std::sort(peaks.begin(), peaks.end(), [](auto& a, auto& b){ return a.v > b.v; });

    // extract segments
    std::vector<cv::Vec4i> result;
    cv::Mat used = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (auto& pk : peaks) {
        double theta = pk.ti * theta_res;
        double rho = pk.ri * rho_res - rhoOff;
        double ct = std::cos(theta), st = std::sin(theta);

        struct LP { int x, y; double proj; };
        std::vector<LP> lp;
        for (auto& p : pts) {
            if (used.at<uchar>(p.y, p.x)) continue;
            if (std::abs(p.x * ct + p.y * st - rho) <= std::max(rho_res, 1.5))
                lp.push_back({p.x, p.y, p.x * (-st) + p.y * ct});
        }
        if (lp.empty()) continue;
        std::sort(lp.begin(), lp.end(), [](auto& a, auto& b){ return a.proj < b.proj; });

        int s = 0;
        for (int i = 1; i <= (int)lp.size(); ++i) {
            bool end = (i == (int)lp.size()) ||
                       (lp[i].proj - lp[i-1].proj > maxGap);
            if (end) {
                double len = std::sqrt(std::pow(lp[i-1].x - lp[s].x, 2.0) +
                                       std::pow(lp[i-1].y - lp[s].y, 2.0));
                if (len >= minLen) {
                    result.push_back({lp[s].x, lp[s].y, lp[i-1].x, lp[i-1].y});
                    for (int j = s; j < i; ++j) used.at<uchar>(lp[j].y, lp[j].x) = 1;
                }
                s = i;
            }
        }
    }
    return result;
}

/* ─── Hough Circles (gradient) ─────────────────────────────────── */

inline std::vector<cv::Vec3f> hough_circles(const cv::Mat& gray, double dp,
    double minDist, double p1, double p2, int minR, int maxR)
{
    int rows = gray.rows, cols = gray.cols;
    cv::Mat blur = gaussian_blur(gray, 5, 1.0);
    cv::Mat edg  = canny(blur, p1 / 2.0, p1, 3);
    cv::Mat gx   = sobel(blur, 1, 0);
    cv::Mat gy   = sobel(blur, 0, 1);

    if (maxR <= 0) maxR = std::max(rows, cols) / 2;
    int aR = (int)(rows / dp + 1), aC = (int)(cols / dp + 1);
    cv::Mat acc = cv::Mat::zeros(aR, aC, CV_32SC1);

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            if (!edg.at<uchar>(y, x)) continue;
            double dx = gx.at<double>(y, x), dy = gy.at<double>(y, x);
            double mg = std::sqrt(dx*dx + dy*dy);
            if (mg < 1e-6) continue;
            dx /= mg; dy /= mg;
            for (int sg = -1; sg <= 1; sg += 2)
                for (int r = minR; r <= maxR; r += 1) {
                    int cx = (int)(x + sg * r * dx), cy = (int)(y + sg * r * dy);
                    int ax = (int)(cx / dp), ay = (int)(cy / dp);
                    if (ax >= 0 && ax < aC && ay >= 0 && ay < aR)
                        acc.at<int>(ay, ax)++;
                }
        }

    struct CC { int x, y, v; };
    std::vector<CC> cands;
    for (int y = 0; y < aR; ++y)
        for (int x = 0; x < aC; ++x) {
            int v = acc.at<int>(y, x);
            if (v < (int)p2) continue;
            bool mx = true;
            for (int dy2 = -1; dy2 <= 1 && mx; ++dy2)
                for (int dx2 = -1; dx2 <= 1 && mx; ++dx2) {
                    if (!dy2 && !dx2) continue;
                    int ny = y+dy2, nx = x+dx2;
                    if (ny>=0&&ny<aR&&nx>=0&&nx<aC && acc.at<int>(ny,nx)>v) mx=false;
                }
            if (mx) cands.push_back({(int)(x*dp+dp/2),(int)(y*dp+dp/2),v});
        }
    std::sort(cands.begin(), cands.end(), [](auto&a,auto&b){return a.v>b.v;});

    std::vector<cv::Vec3f> circles;
    for (auto& c : cands) {
        bool skip = false;
        for (auto& o : circles) {
            double d = std::sqrt((c.x-o[0])*(c.x-o[0])+(c.y-o[1])*(c.y-o[1]));
            if (d < minDist) { skip=true; break; }
        }
        if (skip) continue;
        std::vector<int> rv(maxR+1, 0);
        for (int y=0;y<rows;++y)
            for (int x=0;x<cols;++x) {
                if (!edg.at<uchar>(y,x)) continue;
                int r=(int)(std::sqrt((double)(x-c.x)*(x-c.x)+(double)(y-c.y)*(y-c.y))+0.5);
                if (r>=minR&&r<=maxR) rv[r]++;
            }
        int br=minR, bv=0;
        for (int r=minR;r<=maxR;++r) if(rv[r]>bv){bv=rv[r];br=r;}
        if (bv>0) circles.push_back(cv::Vec3f((float)c.x,(float)c.y,(float)br));
    }
    return circles;
}

/* ─── Find Contours (border following) ─────────────────────────── */

inline std::vector<std::vector<cv::Point>> find_contours(const cv::Mat& bin) {
    int rows = bin.rows, cols = bin.cols;
    cv::Mat vis = cv::Mat::zeros(rows, cols, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    int DX[]={1,1,0,-1,-1,-1,0,1}, DY[]={0,-1,-1,-1,0,1,1,1};

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            if (!bin.at<uchar>(y,x) || vis.at<uchar>(y,x)) continue;
            bool border = (x==0||y==0||x==cols-1||y==rows-1);
            if (!border)
                for (int d=0;d<8&&!border;++d) {
                    int nx=x+DX[d],ny=y+DY[d];
                    if (!bin.at<uchar>(ny,nx)) border=true;
                }
            if (!border) continue;

            std::vector<cv::Point> c;
            int cx=x, cy=y, dir=0;
            // find initial backtrack dir
            for (int d=0;d<8;++d) {
                int nx=cx+DX[d],ny=cy+DY[d];
                if (nx<0||ny<0||nx>=cols||ny>=rows||!bin.at<uchar>(ny,nx))
                    { dir=(d+1)%8; break; }
            }
            int sx=cx, sy=cy; bool first=true;
            int maxSteps = rows*cols;
            do {
                if (first || !(cx==sx&&cy==sy)) {
                    c.push_back({cx,cy}); vis.at<uchar>(cy,cx)=1; first=false;
                }
                bool found=false;
                for (int i=0;i<8;++i) {
                    int d=(dir+i)%8, nx=cx+DX[d], ny=cy+DY[d];
                    if (nx>=0&&ny>=0&&nx<cols&&ny<rows&&bin.at<uchar>(ny,nx)) {
                        cx=nx; cy=ny; dir=(d+5)%8; found=true; break;
                    }
                }
                if (!found) break;
                if (--maxSteps <= 0) break;
            } while (cx!=sx||cy!=sy);
            if (c.size()>=3) contours.push_back(c);
        }
    return contours;
}

/* ─── Fit Ellipse (moment-based) ───────────────────────────────── */

inline cv::RotatedRect fit_ellipse(const std::vector<cv::Point>& pts) {
    int n = (int)pts.size();
    double mx=0, my=0;
    for (auto&p:pts){mx+=p.x;my+=p.y;}
    mx/=n; my/=n;
    double m20=0,m02=0,m11=0;
    for (auto&p:pts){double dx=p.x-mx,dy=p.y-my;m20+=dx*dx;m02+=dy*dy;m11+=dx*dy;}
    m20/=n; m02/=n; m11/=n;
    double tr=m20+m02, det=m20*m02-m11*m11;
    double disc=std::sqrt(std::max(0.0,tr*tr/4.0-det));
    double l1=tr/2.0+disc, l2=tr/2.0-disc;
    double ang=0.5*std::atan2(2.0*m11, m20-m02)*180.0/PI;
    double a=2.0*std::sqrt(std::max(0.0,l1));
    double b=2.0*std::sqrt(std::max(0.0,l2));
    return cv::RotatedRect(cv::Point2f((float)mx,(float)my),
                           cv::Size2f((float)(2*a),(float)(2*b)), (float)ang);
}

/* ─── Distance Transform (chamfer 2-pass) ──────────────────────── */

inline cv::Mat distance_transform(const cv::Mat& bin) {
    int rows=bin.rows, cols=bin.cols;
    float INF=(float)(rows+cols);
    cv::Mat d(rows,cols,CV_32FC1);
    for (int y=0;y<rows;++y)
        for (int x=0;x<cols;++x)
            d.at<float>(y,x)=(bin.at<uchar>(y,x)==0)?0.0f:INF;
    float D1=1.0f, D2=1.41421356f;
    // forward
    for (int y=0;y<rows;++y)
        for (int x=0;x<cols;++x) {
            float&v=d.at<float>(y,x);
            if (y>0) v=std::min(v,d.at<float>(y-1,x)+D1);
            if (x>0) v=std::min(v,d.at<float>(y,x-1)+D1);
            if (y>0&&x>0) v=std::min(v,d.at<float>(y-1,x-1)+D2);
            if (y>0&&x<cols-1) v=std::min(v,d.at<float>(y-1,x+1)+D2);
        }
    // backward
    for (int y=rows-1;y>=0;--y)
        for (int x=cols-1;x>=0;--x) {
            float&v=d.at<float>(y,x);
            if (y<rows-1) v=std::min(v,d.at<float>(y+1,x)+D1);
            if (x<cols-1) v=std::min(v,d.at<float>(y,x+1)+D1);
            if (y<rows-1&&x<cols-1) v=std::min(v,d.at<float>(y+1,x+1)+D2);
            if (y<rows-1&&x>0) v=std::min(v,d.at<float>(y+1,x-1)+D2);
        }
    return d;
}

/* ─── Drawing Primitives ───────────────────────────────────────── */

inline void draw_line(cv::Mat& img, cv::Point p1, cv::Point p2,
                      const cv::Scalar& col, int thick)
{
    int x0=p1.x,y0=p1.y,x1=p2.x,y1=p2.y;
    int dx=std::abs(x1-x0), dy=std::abs(y1-y0);
    int sx=(x0<x1)?1:-1, sy=(y0<y1)?1:-1, err=dx-dy;
    int r=std::max(1,thick/2);
    while (true) {
        for (int iy=-r+1;iy<r;++iy)
            for (int ix=-r+1;ix<r;++ix)
                if (ix*ix+iy*iy<r*r)
                    set_pixel_color(img,x0+ix,y0+iy,col);
        if (x0==x1&&y0==y1) break;
        int e2=2*err;
        if (e2>-dy){err-=dy;x0+=sx;}
        if (e2<dx){err+=dx;y0+=sy;}
    }
}

inline void draw_circle(cv::Mat& img, cv::Point cen, int rad,
                        const cv::Scalar& col, int thick)
{
    if (thick < 0) { // filled
        for (int y=-rad;y<=rad;++y)
            for (int x=-rad;x<=rad;++x)
                if (x*x+y*y<=rad*rad)
                    set_pixel_color(img,cen.x+x,cen.y+y,col);
        return;
    }
    int ri=std::max(0,rad-thick/2), ro=rad+(thick+1)/2;
    int ri2=ri*ri, ro2=ro*ro;
    for (int y=-ro;y<=ro;++y)
        for (int x=-ro;x<=ro;++x) {
            int d2=x*x+y*y;
            if (d2>=ri2&&d2<=ro2)
                set_pixel_color(img,cen.x+x,cen.y+y,col);
        }
}

inline void draw_ellipse_rr(cv::Mat& img, const cv::RotatedRect& rr,
                            const cv::Scalar& col, int thick)
{
    int N=360;
    float ar=rr.angle*(float)PI/180.0f;
    float ca=std::cos(ar),sa=std::sin(ar);
    float a=rr.size.width/2.0f, b=rr.size.height/2.0f;
    std::vector<cv::Point> pts(N);
    for (int i=0;i<N;++i) {
        float t=2.0f*(float)PI*i/N;
        float ex=a*std::cos(t), ey=b*std::sin(t);
        pts[i]={(int)std::round(ex*ca-ey*sa+rr.center.x),
                (int)std::round(ex*sa+ey*ca+rr.center.y)};
    }
    for (int i=0;i<N;++i) draw_line(img,pts[i],pts[(i+1)%N],col,thick);
}

inline void draw_polylines(cv::Mat& img, const std::vector<cv::Point>& pts,
                           bool closed, const cv::Scalar& col, int thick)
{
    for (size_t i=0;i+1<pts.size();++i) draw_line(img,pts[i],pts[i+1],col,thick);
    if (closed && pts.size()>1) draw_line(img,pts.back(),pts[0],col,thick);
}

} // namespace custom
