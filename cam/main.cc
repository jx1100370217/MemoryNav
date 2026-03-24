#include "main.h"

class NvJpgenc
{
public:
    NvJpgenc(uint32_t w = 1920, uint32_t h = 1536) : m_width(w), m_height(h)
    {
        m_jpegenc = NvJPEGEncoder::createJPEGEncoder("jpenenc");
        m_jpegenc->setCropRect(0,0,w,h);
        m_out_buf = new unsigned char[w * h * 3 / 2];
    }

    int Encode(int dmabuf, std::vector<uint8_t>& jpgout, int quality = 80)
    {
        unsigned long size = m_width * m_height * 3 / 2;
        if (m_jpegenc->encodeFromFd(dmabuf, JCS_YCbCr, &m_out_buf, size, quality))
        {
            std::cerr << "encodeFromFd faild." << std::endl;
            return -1;
        }
        if (size > m_width * m_height * 3 / 2)
        {
            std::cerr << "error size:" << size << std::endl;
            return -1;
        }
        jpgout.clear();
        jpgout.assign(m_out_buf, m_out_buf + size);
        return 0;
    }

private:
    unsigned char *m_out_buf;
    uint32_t m_width, m_height;
    NvJPEGEncoder *m_jpegenc;
};

typedef struct {
    std::string type;
    std::string direction;
    int fov;
    int width;
    int height;
} VCamParam;

typedef struct {
    std::string name;
    int dev;
    bool downscale;
    bool flip;
    bool filter;
    std::vector<double> T_ic;
    std::vector<double> distortion_coeffs;
    std::vector<double> intrinsics;
    std::vector<VCamParam> vcam;
} CameraParam;

class CameraNode : public rclcpp::Node
{
public:
    CameraNode(std::string name) : Node(name) {
        RCLCPP_INFO(this->get_logger(), "node start: %s", name.c_str());
        init_cuda();
        m_tf_broadcaster = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
        parse_camera_parameters();
    }

    ~CameraNode() { }

private:
    void init_cuda()
    {
        cudaError_t cuda_ret = cudaSetDevice(0);
        if (cuda_ret != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(), "CUDA device set failed: %s", cudaGetErrorString(cuda_ret));
            return;
        }
        RCLCPP_INFO(this->get_logger(), "CUDA initialized successfully");
    }
    
    void parse_camera_parameters()
    {
        std::string params_file;
        std::vector<int> video_dev;
        this->declare_parameter<std::string>("params", "");
        params_file = this->get_parameter("params").as_string();
        YAML::Node config = YAML::LoadFile(params_file);
        auto qos = rclcpp::QoS(10).keep_last(1).reliable().durability_volatile();
        for (const auto& camera : config["cameras"]) {
            std::string camera_name = camera["name"].as<std::string>();
            CameraParam param;
            param.name = camera_name;
            param.dev = camera["dev"].as<int>();
            param.downscale = camera["downscale"].as<bool>();
            param.flip = camera["flip"].as<bool>();
            param.filter = camera["filter"].as<bool>();
            if (camera["T_ic"]) {
                param.T_ic = camera["T_ic"].as<std::vector<double>>();
                publish_camera_transform(camera_name, param.T_ic);
            }
            if (camera["distortion_coeffs"])
                param.distortion_coeffs = camera["distortion_coeffs"].as<std::vector<double>>();
            if (camera["intrinsics"])
                param.intrinsics = camera["intrinsics"].as<std::vector<double>>();
            if (camera["vcam"])
                param.vcam = parse_vcam_parameters(param, camera["vcam"]);
            m_camera_params[camera_name] = param;
            m_camera_name[param.dev] = camera_name;
            m_camera_pub[param.dev] = this->create_publisher<sensor_msgs::msg::Image>(
                camera_name + "/image_raw", qos
            );
            m_camera_jpeg_pub[param.dev] = this->create_publisher<sensor_msgs::msg::CompressedImage>(
                camera_name + "/image_raw/compressed", qos
            );
            m_camera_buff_fd[param.dev] = nullptr;
            NvBufSurf::NvCommonTransformParams transform_param;
            memset(&transform_param, 0, sizeof(transform_param));
            m_camera_transform_params[param.dev] = transform_param;
            m_camera_jpegenc[param.dev] = nullptr;
            m_camera_thread_pool.emplace(param.dev, 1);
            video_dev.push_back(param.dev);
        }
        m_stitch_pub = this->create_publisher<sensor_msgs::msg::Image>(
            "/stitch/image_raw", qos);
        m_video = std::make_shared<Video>(video_dev, std::bind(&CameraNode::camera_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        m_param_update_pub = this->create_publisher<std_msgs::msg::String>(
            "/robot/sensors/params", rclcpp::QoS(10).reliable().transient_local());
        std_msgs::msg::String msg;
        int left, right, width, height;
        get_stitch_params(left, right, width, height);
        msg.data = "left_x:" + std::to_string(left) + ",right_x:" + std::to_string(right) + ",width:" + std::to_string(width) + ",height:" + std::to_string(height);
        m_param_update_pub->publish(msg);
    }

    std::vector<VCamParam> parse_vcam_parameters(const CameraParam& camera_param, const YAML::Node& vcam_node)
    {
        std::vector<VCamParam> vcam_params;
        for (const auto& vcam_param : vcam_node) {
            VCamParam param;
            param.type = vcam_param["type"].as<std::string>();
            param.direction = vcam_param["direction"].as<std::string>();
            param.fov = vcam_param["fov"].as<int>();
            param.width = vcam_param["width"].as<int>();
            param.height = vcam_param["height"].as<int>();
            vcam_params.push_back(param);
            auto fisheye_undist = std::make_shared<FisheyeUndist>(
                camera_param.intrinsics, camera_param.distortion_coeffs,
                param.fov, param.width, param.height, param.direction,
                camera_param.T_ic, param.type, camera_param.downscale);
            auto pub = this->create_publisher<sensor_msgs::msg::Image>(
                camera_param.name + "/" + param.direction + "_" + param.type + "/image_raw",
                rclcpp::QoS(10).keep_last(1).reliable().durability_volatile()
            );
            m_camera_undist[camera_param.dev].push_back(std::make_pair(fisheye_undist, pub));
        }
        return vcam_params;
    }

private:

    std::shared_ptr<uint8_t> convert2rgb(int dev, int dmabuf, uint32_t& pitch, uint32_t& width, uint32_t& height)
    {
        if (m_camera_buff_fd[dev] == nullptr) {
            m_camera_buff_fd[dev] = new int[2];
            GetDmaPtr(dmabuf, pitch, width, height);
            NvBufSurf::NvCommonAllocateParams param;
            param.width = m_camera_params[m_camera_name[dev]].downscale ? width / 2 : width;
            param.height = m_camera_params[m_camera_name[dev]].downscale ? height / 2 : height;
            param.layout = NVBUF_LAYOUT_PITCH;
            param.memType = NVBUF_MEM_SURFACE_ARRAY;
            param.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
            param.memtag = NvBufSurfaceTag_VIDEO_CONVERT;
            NvBufSurf::NvAllocate(&param, 1, m_camera_buff_fd[dev]);
            param.colorFormat = NVBUF_COLOR_FORMAT_RGB;
            NvBufSurf::NvAllocate(&param, 1, m_camera_buff_fd[dev] + 1);
            m_camera_transform_params[dev].src_top = 0;
            m_camera_transform_params[dev].src_left = 0;
            m_camera_transform_params[dev].src_width = width;
            m_camera_transform_params[dev].src_height = height;
            m_camera_transform_params[dev].dst_top = 0;
            m_camera_transform_params[dev].dst_left = 0;
            m_camera_transform_params[dev].dst_width = param.width;
            m_camera_transform_params[dev].dst_height = param.height;
            m_camera_transform_params[dev].flag = (NvBufSurfTransform_Transform_Flag)(NVBUFSURF_TRANSFORM_FILTER);
            m_camera_transform_params[dev].filter = NvBufSurfTransformInter_Nearest;
            if (m_camera_params[m_camera_name[dev]].flip) {
                m_camera_transform_params[dev].flag = (NvBufSurfTransform_Transform_Flag)(NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_FLIP);
                m_camera_transform_params[dev].flip = NvBufSurfTransform_Rotate180;
            }
            m_camera_jpegenc[dev] = std::make_shared<NvJpgenc>(param.width, param.height);
        }
        NvBufSurfTransformConfigParams config_params;
        config_params.compute_mode = NvBufSurfTransformCompute_VIC;
        NvBufSurfTransformSetSessionParams(&config_params);
        NvBufSurf::NvTransform(&m_camera_transform_params[dev], dmabuf, m_camera_buff_fd[dev][0]);
        auto tparam = m_camera_transform_params[dev];
        tparam.src_width = tparam.dst_width;
        tparam.src_height = tparam.dst_height;
        tparam.flag = (NvBufSurfTransform_Transform_Flag)(NVBUFSURF_TRANSFORM_FILTER);
        tparam.filter = NvBufSurfTransformInter_Nearest;
        config_params.gpu_id = 0;
        config_params.compute_mode = NvBufSurfTransformCompute_GPU;
        cudaStreamCreateWithFlags(&(config_params.cuda_stream), cudaStreamNonBlocking);
        NvBufSurfTransformSetSessionParams(&config_params);
        NvBufSurf::NvTransform(&tparam, m_camera_buff_fd[dev][0], m_camera_buff_fd[dev][1]);
        cudaStreamSynchronize(config_params.cuda_stream);
        cudaStreamDestroy(config_params.cuda_stream);
        return GetDmaPtr(m_camera_buff_fd[dev][1], pitch, width, height);
    }

    void get_stitch_params(int& left, int& right, int& width, int& height)
    {
        const int rwidth = 640;
        left = static_cast<int>(rwidth * m_left_x);
        right = static_cast<int>(left * m_right_x);
        width = left + (rwidth - right);
        width = ALIGN_UP(width, 16);
        height = 480;
    }

    std::shared_ptr<cv::cuda::GpuMat> stitchImages(const cv::cuda::GpuMat& img1, 
        const cv::cuda::GpuMat& img2)
    {
        int width = img1.cols;
        int height = img1.rows;
        int nleft, nright, stitched_width, stitched_height;
        get_stitch_params(nleft, nright, stitched_width, stitched_height);

        cv::Rect left_roi(0, 0, nleft, height);
        cv::cuda::GpuMat left_half(img1, left_roi);

        cv::Rect right_roi(nright, 0, width - nright, height);
        cv::cuda::GpuMat right_half(img2, right_roi);

        std::shared_ptr<cv::cuda::GpuMat> stitched = std::make_shared<cv::cuda::GpuMat>(
        height, stitched_width, img1.type());

        cv::Rect left_dst_roi(0, 0, left_half.cols, height);
        cv::cuda::GpuMat left_dst_roi_mat(*stitched, left_dst_roi);
        left_half.copyTo(left_dst_roi_mat);

        cv::Rect right_dst_roi(left_half.cols, 0, right_half.cols, height);
        cv::cuda::GpuMat right_dst_roi_mat(*stitched, right_dst_roi);
        right_half.copyTo(right_dst_roi_mat);

        return stitched;
    }

    void camera_callback(int dev, int buff_fd, uint64_t timestamp)
    {
        uint32_t pitch, width, height;
        auto header = std_msgs::msg::Header();
        struct timespec tp;
        clock_gettime(CLOCK_MONOTONIC, &tp);
        long long nowtsus = tp.tv_sec * 1000000 + tp.tv_nsec / 1000 - timestamp;
        header.stamp = this->now() - rclcpp::Duration::from_seconds(nowtsus / 1e6);
        header.frame_id = m_camera_name[dev];
        auto ptr = convert2rgb(dev, buff_fd, pitch, width, height);
        cv::Mat image(height, width, CV_8UC3, ptr.get(), pitch);
        if (m_camera_undist.count(dev) > 0) {
            for (const auto& [fisheye_undist, pub] : m_camera_undist[dev]) {
                // if (pub->get_subscription_count() == 0)
                //     continue;
                auto nheader = header;
                nheader.frame_id = header.frame_id + "_" + fisheye_undist->get_name();
                std::shared_ptr<cv::Mat> img = nullptr;
                if (m_stitch_img.count(nheader.frame_id) > 0) {
                    img = fisheye_undist->remap(image, m_stitch_img[nheader.frame_id]);
                    if (header.frame_id == "camera_1" && !m_stitch_img["camera_2_front_cylindrical"]->empty()) {
                        auto& img_right = m_stitch_img["camera_2_front_cylindrical"];
                        auto& img_left = m_stitch_img[nheader.frame_id];
                        auto stitched = stitchImages(*img_left, *img_right);
                        cv::Mat stitched_mat;
                        stitched->download(stitched_mat);
                        auto sheader = header;
                        sheader.frame_id = "stitch";
                        m_stitch_pub->publish(*cv_bridge::CvImage(sheader, "rgb8", stitched_mat).toImageMsg());
                    }
                } else
                    img = fisheye_undist->remap(image);
                pub->publish(*cv_bridge::CvImage(nheader, "rgb8", *img).toImageMsg());
            }
        }
        if (m_camera_pub[dev]->get_subscription_count() > 0)
            m_camera_pub[dev]->publish(*cv_bridge::CvImage(header, "rgb8", image).toImageMsg());
        if (m_camera_thread_pool[dev].n_idle() && m_camera_jpeg_pub[dev]->get_subscription_count() > 0) {
            m_camera_thread_pool[dev].push(std::bind([this, dev, header] {
                std::vector<uint8_t> jpgout;
                m_camera_jpegenc[dev]->Encode(m_camera_buff_fd[dev][0], jpgout);
                auto msg = sensor_msgs::msg::CompressedImage();
                msg.header = header;
                msg.format = "jpeg";
                msg.data = jpgout;
                m_camera_jpeg_pub[dev]->publish(msg);
            }));
        }
    }

    std::shared_ptr<uint8_t> GetDmaPtr(int dmabuf, uint32_t& pitch, uint32_t& width, uint32_t& height)
    {
        NvBufSurface *nvbuf_surf = 0;
        NvBufSurfaceFromFd(dmabuf, (void**)(&nvbuf_surf));
        NvBufSurfaceMap(nvbuf_surf, 0, 0, NVBUF_MAP_READ);
        NvBufSurfaceSyncForCpu (nvbuf_surf, 0, 0);
        pitch = nvbuf_surf->surfaceList->planeParams.pitch[0];
        width = nvbuf_surf->surfaceList->planeParams.width[0];
        height = nvbuf_surf->surfaceList->planeParams.height[0];
        return std::shared_ptr<uint8_t>((uint8_t*)nvbuf_surf->surfaceList->mappedAddr.addr[0],
            [nvbuf_surf](uint8_t* ptr){
                NvBufSurfaceSyncForDevice(nvbuf_surf, 0, 0);
                NvBufSurfaceUnMap(nvbuf_surf, 0, 0);
            });
    }

    void publish_camera_transform(const std::string& camera_name, const std::vector<double>& T_ic)
    {
        if (T_ic.empty() || T_ic.back() < 0.0)
            return;
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = this->now();
        transform.header.frame_id = "imu_frame";
        transform.child_frame_id = camera_name;
        transform.transform.translation.x = T_ic[3];
        transform.transform.translation.y = T_ic[7];
        transform.transform.translation.z = T_ic[11];
        tf2::Quaternion q;
        tf2::Matrix3x3 rot;
        rot.setValue(
            T_ic[0], T_ic[1], T_ic[2],
            T_ic[4], T_ic[5], T_ic[6],
            T_ic[8], T_ic[9], T_ic[10]
        );
        rot.getRotation(q);
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        transform.transform.rotation.w = q.w();
        m_tf_broadcaster->sendTransform(transform);
        RCLCPP_INFO(this->get_logger(), "Published static transform for %s", camera_name.c_str());
    }

    std::map<std::string, CameraParam> m_camera_params;
    std::shared_ptr<Video> m_video;
    std::map<int, std::string> m_camera_name;
    std::map<int, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> m_camera_pub;
    std::map<int, rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr> m_camera_jpeg_pub;
    std::map<int, int*> m_camera_buff_fd;
    std::map<int, NvBufSurf::NvCommonTransformParams> m_camera_transform_params;
    std::map<int, std::shared_ptr<NvJpgenc>> m_camera_jpegenc;
    std::map<int, ctpl::thread_pool> m_camera_thread_pool;
    std::map<int, std::vector<std::pair<std::shared_ptr<FisheyeUndist>,
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>>> m_camera_undist;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> m_tf_broadcaster;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_param_update_pub;
    // stitch
    std::map<std::string, std::shared_ptr<cv::cuda::GpuMat>> m_stitch_img = {
        {"camera_1_front_cylindrical", std::make_shared<cv::cuda::GpuMat>()},
        {"camera_2_front_cylindrical", std::make_shared<cv::cuda::GpuMat>()}
    };
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_stitch_pub;
    double m_left_x = 0.78, m_right_x = 0.98;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<CameraNode>("robot_sensors");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}