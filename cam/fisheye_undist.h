#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cmath>
#include <Eigen/Dense>

#define RAW_WIDTH 1920
#define RAW_HEIGHT 1536

#define DEG_TO_RAD (M_PI / 180.0)

class FisheyeUndist {
public:
    FisheyeUndist(const std::vector<double>& intrinsics,
                  const std::vector<double>& dist_coeffs,
                  int fov, int width, int height,
                  const std::string direction,
                  const std::vector<double>& extrinsic, // cam to imu
                  const std::string mode="pinhole", bool downscale=false)
        : m_direction(direction), m_mode(mode), m_fov(fov), m_width(width), m_height(height)
    {
        cv::cuda::setDevice(0);
        m_intrinsics = Eigen::Map<const Eigen::VectorXd>(intrinsics.data(), intrinsics.size()).transpose(); // imu to cam
        m_dist_coeffs = Eigen::Map<const Eigen::VectorXd>(dist_coeffs.data(), dist_coeffs.size());
        m_extrinsic = Eigen::Map<const Eigen::Matrix4d>(extrinsic.data());
        if (mode == "pinhole")
            generateUndistMapPinhole();
        else if (mode == "cylindrical")
            generateUndistMapCylindrical();
    }
    ~FisheyeUndist() {}

    void generateUndistMapPinhole()
    {
        double center_x = (double)m_width / 2;
        double center_y = (double)m_height / 2;
        double focal_gen = (m_width / 2) / std::tan(m_fov * DEG_TO_RAD / 2);
        Eigen::Matrix3d rot = m_extrinsic.block<3, 3>(0, 0) * m_direction_matrix.at(m_direction);
        cv::Mat xmap = cv::Mat::zeros(m_height, m_width, CV_32F);
        cv::Mat ymap = cv::Mat::zeros(m_height, m_width, CV_32F);

        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                Eigen::Vector3d objPoint = rot * Eigen::Vector3d(((double)x - center_x), ((double)y - center_y), focal_gen);
                Eigen::Vector2d imgPoint;
                projectPoints(objPoint, imgPoint);
                xmap.at<float>(y, x) = imgPoint.x();
                ymap.at<float>(y, x) = imgPoint.y();
            }
        }
        m_undist_map = std::make_pair(cv::cuda::GpuMat(xmap), cv::cuda::GpuMat(ymap));
    }

    void generateUndistMapCylindrical()
    {
        double center_x = (double)m_width / 2;
        double center_y = (double)m_height / 2;
        double focal_gen = m_width / (m_fov * DEG_TO_RAD);
        Eigen::Matrix3d rot = m_extrinsic.block<3, 3>(0, 0) * m_direction_matrix.at(m_direction);
        cv::Mat xmap = cv::Mat::zeros(m_height, m_width, CV_32F);
        cv::Mat ymap = cv::Mat::zeros(m_height, m_width, CV_32F);

        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                double nx = std::tan((x - m_width / 2) / focal_gen) * focal_gen;
                double ny = (y - m_height / 2) * std::sqrt(nx*nx + focal_gen*focal_gen) / focal_gen;
                Eigen::Vector3d objPoint = rot * Eigen::Vector3d(nx, ny, focal_gen);
                Eigen::Vector2d imgPoint;
                projectPoints(objPoint, imgPoint);
                xmap.at<float>(y, x) = imgPoint.x();
                ymap.at<float>(y, x) = imgPoint.y();
            }
        }
        m_undist_map = std::make_pair(cv::cuda::GpuMat(xmap), cv::cuda::GpuMat(ymap));
    }

    std::shared_ptr<cv::Mat> remap(const cv::Mat& img)
    {
        cv::cuda::GpuMat img_cuda(img), img_cuda_remap;
        std::shared_ptr<cv::Mat> output = std::make_shared<cv::Mat>();
        cv::cuda::remap(img_cuda, img_cuda_remap, m_undist_map.first, m_undist_map.second, cv::INTER_LINEAR);
        img_cuda_remap.download(*output);
        return output;
    }

    std::shared_ptr<cv::Mat> remap(const cv::Mat& img, std::shared_ptr<cv::cuda::GpuMat> img_cuda_remap)
    {
        cv::cuda::GpuMat img_cuda(img);
        std::shared_ptr<cv::Mat> output = std::make_shared<cv::Mat>();
        cv::cuda::remap(img_cuda, *img_cuda_remap, m_undist_map.first, m_undist_map.second, cv::INTER_LINEAR);
        img_cuda_remap->download(*output);
        return output;
    }

    std::string get_name() const
    {
        return m_direction + "_" + m_mode;
    }

private:

    void projectPoints(const Eigen::Vector3d& objPoint, Eigen::Vector2d& imgPoint)
    {
        auto xs = objPoint / objPoint.norm();
        Eigen::Vector2d xu, xd;
        xu(0) = xs(0) / (xs(2) + m_intrinsics(0));
        xu(1) = xs(1) / (xs(2) + m_intrinsics(0));
        double r2 = xu.dot(xu);
        double r4 = r2 * r2;
        xd[0] = xu[0] * (1 + m_dist_coeffs(0) * r2 + m_dist_coeffs(1) * r4) + 2 * m_dist_coeffs(2) * xu[0] * xu[1] + m_dist_coeffs(3) * (r2 + 2 * xu[0] * xu[0]);
        xd[1] = xu[1] * (1 + m_dist_coeffs(0) * r2 + m_dist_coeffs(1) * r4) + m_dist_coeffs(2) * (r2 + 2 * xu[1] * xu[1]) + 2 * m_dist_coeffs(3) * xu[0] * xu[1];
        imgPoint[0] = m_intrinsics(1) * xd[0] + m_intrinsics(3);
        imgPoint[1] = m_intrinsics(2) * xd[1] + m_intrinsics(4);
    }

    std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat> m_undist_map;
    std::pair<cv::Mat, cv::Mat> m_undist_map_cpu;
    Eigen::VectorXd m_intrinsics, m_dist_coeffs;
    Eigen::Matrix4d m_extrinsic;
    const std::map<std::string, Eigen::Matrix3d> m_direction_matrix = {
        {"front", (Eigen::Matrix3d() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished()},
        {"right", (Eigen::Matrix3d() << -1, 0, 0, 0, 0, -1, 0, -1, 0).finished()},
        {"back",  (Eigen::Matrix3d() << 0, 0, -1, 1, 0, 0, 0, -1, 0).finished()},
        {"left",  (Eigen::Matrix3d() << 1, 0, 0, 0, 0, 1, 0, -1, 0).finished()}
    };
    std::string m_direction;
    std::string m_mode;
    int m_fov;
    int m_width;
    int m_height;
};