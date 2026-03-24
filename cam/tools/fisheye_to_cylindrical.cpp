/**
 * 鱼眼图像转柱面视角工具
 * 参考 cam 仓库的 FisheyeUndist 实现
 *
 * 用法:
 *   fisheye_to_cylindrical -i input.jpg -o output.jpg -c params.yaml
 *   fisheye_to_cylindrical -i input.jpg -o output.jpg -c params.yaml -n camera_1
 *   fisheye_to_cylindrical -i input.jpg -o output.jpg --intrinsics "..." --distortion "..."
 */

#include "fisheye_undist_cpu.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <vector>

static void printUsage(const char* prog) {
    std::cerr << "用法: " << prog << " [选项]\n"
              << "  -i, --input <path>      输入鱼眼图像路径\n"
              << "  -o, --output <path>     输出柱面图路径\n"
              << "  -c, --config <path>     相机参数 YAML（仅读取 intrinsics/distortion_coeffs）\n"
              << "  -n, --name <name>       相机名称，从 config 选取\n"
              << "  -f, --fov <deg>         输出视场角，默认 180\n"
              << "  -w, --width <px>        输出宽度，默认 640\n"
              << "  -h, --height <px>       输出高度，默认 480\n"
              << "  -p, --pitch-up <deg>    视角向上偏移角度，默认 0\n"
              << "  --intrinsics <vals>      内参 [xi,fx,fy,cx,cy]，逗号分隔\n"
              << "  --distortion <vals>     畸变 [k1,k2,p1,p2]，逗号分隔\n"
              << "\n示例:\n"
              << "  " << prog << " -i fisheye.jpg -o cyl.jpg -c params.yaml -n camera_1\n"
              << "  " << prog << " -i fisheye.jpg -o cyl.jpg --intrinsics \"1.77,1238,1238,962,767\" --distortion \"1.43,-1.78,-0.001,0.005\"\n";
}

static bool parseDoubleList(const std::string& s, std::vector<double>& out) {
    out.clear();
    std::string t = s;
    size_t pos;
    while ((pos = t.find(',')) != std::string::npos) {
        out.push_back(std::stod(t.substr(0, pos)));
        t = t.substr(pos + 1);
    }
    if (!t.empty())
        out.push_back(std::stod(t));
    return !out.empty();
}

static bool loadCameraFromYaml(const std::string& configPath, const std::string& cameraName,
                               std::vector<double>& intrinsics, std::vector<double>& distortion) {
    YAML::Node config = YAML::LoadFile(configPath);
    if (!config["cameras"]) {
        std::cerr << "config 中缺少 cameras 节点\n";
        return false;
    }
    for (const auto& cam : config["cameras"]) {
        std::string name = cam["name"].as<std::string>("");
        if (!cameraName.empty() && name != cameraName)
            continue;
        if (!cam["intrinsics"] || !cam["distortion_coeffs"]) {
            if (cameraName.empty())
                continue;
            std::cerr << "相机 " << name << " 缺少 intrinsics 或 distortion_coeffs\n";
            return false;
        }
        intrinsics = cam["intrinsics"].as<std::vector<double>>();
        distortion = cam["distortion_coeffs"].as<std::vector<double>>();
        return true;
    }
    std::cerr << "未找到相机配置: " << (cameraName.empty() ? "(第一个有内参的)" : cameraName) << "\n";
    return false;
}

int main(int argc, char** argv) {
    std::string inputPath, outputPath, configPath, cameraName;
    std::string intrinsicsStr, distortionStr;
    int fov = 180, outWidth = 640, outHeight = 480;
    double pitchUp = 0.0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc)
            inputPath = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
            outputPath = argv[++i];
        else if ((arg == "-c" || arg == "--config") && i + 1 < argc)
            configPath = argv[++i];
        else if ((arg == "-n" || arg == "--name") && i + 1 < argc)
            cameraName = argv[++i];
        else if ((arg == "-f" || arg == "--fov") && i + 1 < argc)
            fov = std::stoi(argv[++i]);
        else if ((arg == "-w" || arg == "--width") && i + 1 < argc)
            outWidth = std::stoi(argv[++i]);
        else if ((arg == "-h" || arg == "--height") && i + 1 < argc)
            outHeight = std::stoi(argv[++i]);
        else if ((arg == "-p" || arg == "--pitch-up") && i + 1 < argc)
            pitchUp = std::stod(argv[++i]);
        else if (arg == "--intrinsics" && i + 1 < argc)
            intrinsicsStr = argv[++i];
        else if (arg == "--distortion" && i + 1 < argc)
            distortionStr = argv[++i];
        else if (arg == "--help" || arg == "-?") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (inputPath.empty() || outputPath.empty()) {
        std::cerr << "错误: 必须指定 -i 和 -o\n";
        printUsage(argv[0]);
        return 1;
    }

    std::vector<double> intrinsics, distortion;
    if (!configPath.empty()) {
        if (!loadCameraFromYaml(configPath, cameraName, intrinsics, distortion)) {
            return 1;
        }
    }
    if (!intrinsicsStr.empty()) {
        if (!parseDoubleList(intrinsicsStr, intrinsics) || intrinsics.size() != 5) {
            std::cerr << "错误: intrinsics 需 5 个值 [xi,fx,fy,cx,cy]\n";
            return 1;
        }
    }
    if (!distortionStr.empty()) {
        if (!parseDoubleList(distortionStr, distortion) || distortion.size() != 4) {
            std::cerr << "错误: distortion 需 4 个值 [k1,k2,p1,p2]\n";
            return 1;
        }
    }

    if (intrinsics.size() != 5 || distortion.size() != 4) {
        std::cerr << "错误: 缺少相机内参或畸变参数，请使用 -c 或 --intrinsics/--distortion\n";
        return 1;
    }

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "错误: 无法读取图像 " << inputPath << "\n";
        return 1;
    }

    if (img.channels() == 1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    try {
        FisheyeUndistCpu undist(intrinsics, distortion, fov, outWidth, outHeight, pitchUp);
        cv::Mat result = undist.remap(img);
        if (!cv::imwrite(outputPath, result)) {
            std::cerr << "错误: 无法写入 " << outputPath << "\n";
            return 1;
        }
        std::cout << "已保存: " << outputPath << " (" << result.cols << "x" << result.rows << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
