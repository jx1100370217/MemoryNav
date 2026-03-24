#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <string>

#define DEG_TO_RAD (M_PI / 180.0)

/**
 * 鱼眼去畸变 - CPU 版本（无 CUDA 依赖）
 * 柱面投影时视角不变换，直接使用相机坐标系
 */
class FisheyeUndistCpu {
public:
    /**
     * @param intrinsics: [xi, fx, fy, cx, cy] 鱼眼内参
     * @param dist_coeffs: [k1, k2, p1, p2] 畸变系数
     * @param fov: 输出柱面视角的视场角（度）
     * @param width, height: 输出图像尺寸
     * @param pitch_up: 视角向上偏移角度（度），正数表示向上，默认 0
     */
    FisheyeUndistCpu(const std::vector<double>& intrinsics,
                     const std::vector<double>& dist_coeffs,
                     int fov, int width, int height,
                     double pitch_up = 0.0)
        : m_fov(fov), m_width(width), m_height(height), m_pitch_up(pitch_up)
    {
        m_intrinsics = Eigen::Map<const Eigen::VectorXd>(intrinsics.data(), intrinsics.size());
        m_dist_coeffs = Eigen::Map<const Eigen::VectorXd>(dist_coeffs.data(), dist_coeffs.size());
        generateUndistMapCylindrical();
    }

    cv::Mat remap(const cv::Mat& img) const {
        cv::Mat output;
        cv::remap(img, output, m_xmap, m_ymap, cv::INTER_LINEAR);
        return output;
    }

private:
    void projectPoints(const Eigen::Vector3d& objPoint, Eigen::Vector2d& imgPoint) const {
        auto xs = objPoint / objPoint.norm();
        Eigen::Vector2d xu, xd;
        xu(0) = xs(0) / (xs(2) + m_intrinsics(0));
        xu(1) = xs(1) / (xs(2) + m_intrinsics(0));
        double r2 = xu.dot(xu);
        double r4 = r2 * r2;
        xd[0] = xu[0] * (1 + m_dist_coeffs(0) * r2 + m_dist_coeffs(1) * r4)
            + 2 * m_dist_coeffs(2) * xu[0] * xu[1]
            + m_dist_coeffs(3) * (r2 + 2 * xu[0] * xu[0]);
        xd[1] = xu[1] * (1 + m_dist_coeffs(0) * r2 + m_dist_coeffs(1) * r4)
            + m_dist_coeffs(2) * (r2 + 2 * xu[1] * xu[1])
            + 2 * m_dist_coeffs(3) * xu[0] * xu[1];
        imgPoint[0] = m_intrinsics(1) * xd[0] + m_intrinsics(3);
        imgPoint[1] = m_intrinsics(2) * xd[1] + m_intrinsics(4);
    }

    void generateUndistMapCylindrical() {
        double focal_gen = m_width / (m_fov * DEG_TO_RAD);
        double theta = m_pitch_up * DEG_TO_RAD;
        double c = std::cos(theta), s = std::sin(theta);
        m_xmap = cv::Mat::zeros(m_height, m_width, CV_32F);
        m_ymap = cv::Mat::zeros(m_height, m_width, CV_32F);

        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                double nx = std::tan((x - m_width / 2) / focal_gen) * focal_gen;
                double ny = (y - m_height / 2) * std::sqrt(nx * nx + focal_gen * focal_gen) / focal_gen;
                // 绕 x 轴旋转 pitch_up 度，实现视角向上偏移
                Eigen::Vector3d objPoint(nx, ny * c - focal_gen * s, ny * s + focal_gen * c);
                Eigen::Vector2d imgPoint;
                projectPoints(objPoint, imgPoint);
                m_xmap.at<float>(y, x) = (float)imgPoint.x();
                m_ymap.at<float>(y, x) = (float)imgPoint.y();
            }
        }
    }

    cv::Mat m_xmap, m_ymap;
    Eigen::VectorXd m_intrinsics, m_dist_coeffs;
    int m_fov;
    double m_pitch_up;
    int m_width;
    int m_height;
};
