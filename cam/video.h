#pragma once

#include <map>
#include "../src/ctpl_stl.h"

#define FRONT_WIDTH 3840
#define FRONT_HEIGHT 2160
#define VIDEO_WIDTH 1920
#define VIDEO_HEIGHT 1536
#define BUFFER_NUM 10

class TscOffset
{
public:
    TscOffset(const TscOffset&) = delete;
    TscOffset& operator=(const TscOffset&) = delete;
    static TscOffset& getInstance()
    {
        static TscOffset instance;
        return instance;
    }

    // uint64_t GetTscOffset() const { return m_tscoffset.load(); }

    uint64_t GetTscOffset()
    {
        unsigned long raw_nsec, tsc_ns;
        unsigned long cycles, frq;
        struct timespec tp;

        asm volatile("mrs %0, cntfrq_el0" : "=r"(frq));
        asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));

        clock_gettime(CLOCK_MONOTONIC, &tp);

        tsc_ns = (cycles * 100 / (frq / 10000)) * 1000;
        raw_nsec = tp.tv_sec * 1000000000 + tp.tv_nsec;

        return llabs(tsc_ns-raw_nsec) / 1000;
    }

    uint64_t GetNowTscNs()
    {
        unsigned long tsc_ns;
        unsigned long cycles, frq;

        asm volatile("mrs %0, cntfrq_el0" : "=r"(frq));
        asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));

        tsc_ns = (cycles * 1000000000 / frq);
        return tsc_ns;
    }

private:
    TscOffset()
    {
        m_thread = std::thread([this]{
            while (1)
            {
                std::ifstream file("/sys/devices/system/clocksource/clocksource0/offset_ns"); // 打开文件
                std::string line;
                if (std::getline(file, line))
                {
                    line.erase(line.length() - 3);
                    this->m_tscoffset.store(std::stoi(line));
                }
                file.close();
                sleep(1);
            }
        });
    }
    std::atomic<uint64_t> m_tscoffset{0};
    std::thread m_thread;
};

class Video
{
public:
    Video(std::vector<int> dev, std::function<void(int, int, uint64_t)> callback = nullptr) : m_callback(callback)
    {
        std::ifstream file("/tmp/camera_dev.txt");
        std::string filenames;
        std::vector<std::string> m_devs;
        while (std::getline(file, filenames))
        {
            std::istringstream iss(filenames);
            std::string filename;
            while (iss >> filename) {
                m_devs.push_back(filename);
            }
            if (m_devs.size() >= 8)
                break;
        }
        file.close();
        for (auto& d : dev)
        {
            int fd = open(m_devs[d].c_str(), O_RDWR);
            Setup(fd, d == dev[0]);
            m_process.push_back(std::thread(&Video::Process, this, fd, d));            
        }
    }

    virtual void GetRaw(int dev, int fd, uint64_t us)
    {
        // printf("get jpg dev:%d, jpg size:%ld, ts:%ld.%06ld\r\n", dev, jpg.size(), ts.tv_sec, ts.tv_usec);
    }

private:

    static bool GetPtsRemain(uint64_t pts)
    {
        static uint64_t s_pts = 0, s_frame = 0, s_force = 1;
        static std::mutex s_lock;
        uint32_t ret;
        std::lock_guard<std::mutex> lock(s_lock);
        if (pts > s_pts && pts - s_pts > 20000)
        {
            s_frame++;
            s_pts = pts;
            ret = (s_pts / 1000) % 100;
            if ((s_frame >= 150 || s_force) && ret >=35 && ret <= 70)
                s_frame = s_force = 0;
        }
        if (pts + 10000 < s_pts)
        {
            std::cout << "delay frame pts: " << pts << "us, spts: " << s_pts << "us, gap:" << s_pts - pts << "us" << std::endl;
            return false;
        }
        return (s_frame % 3 == 0) ? true : false;
    }

    void Process(int fd, int dev)
    {
        ctpl::thread_pool tpool(1);
        struct pollfd fds[1];
        fds[0].fd = fd;
        fds[0].events = POLLIN;

        while (poll(fds, 1, 5000) > 0 && !m_quit)
        {
            if (!(fds[0].revents & POLLIN))
                continue;
            std::shared_ptr<struct v4l2_buffer> v4l2_buf(new (struct v4l2_buffer),
                [fd](struct v4l2_buffer *ptr){
                    ioctl(fd, VIDIOC_QBUF, ptr);
                    delete ptr;
                });

            /* Dequeue a camera buff */
            memset(v4l2_buf.get(), 0, sizeof(v4l2_buf));
            v4l2_buf->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l2_buf->memory = V4L2_MEMORY_DMABUF;
            if (ioctl(fd, VIDIOC_DQBUF, v4l2_buf.get()) < 0)
            {
                std::cerr << "get camera faild video" <<  dev << std::endl;
                continue;
            }
            if (v4l2_buf->timestamp.tv_sec == 0 && v4l2_buf->timestamp.tv_usec == 0)
                continue;
            uint64_t timestamp = v4l2_buf->timestamp.tv_sec *  1000000 + v4l2_buf->timestamp.tv_usec
                                    - TscOffset::getInstance().GetTscOffset();
            if (GetPtsRemain(timestamp))
            {
                tpool.push(std::bind([this, fd, v4l2_buf, dev, timestamp] {
                    if (m_callback != nullptr)
                        m_callback(dev, m_videodma[fd][v4l2_buf->index], timestamp);
                    else
                        GetRaw(dev, m_videodma[fd][v4l2_buf->index], timestamp);
                }));
            }
        }
    }

    void Setup(int fd, bool front = false)
    {
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = front ? FRONT_WIDTH : VIDEO_WIDTH;
        fmt.fmt.pix.height = front ? FRONT_HEIGHT : VIDEO_HEIGHT;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_UYVY;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0)
        {
            perror("Setting Pixel Format");
            close(fd);
            return ;
        }
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = BUFFER_NUM;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_DMABUF;
        if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0)
        {
            perror("Requesting Buffer");
            close(fd);
            return ;
        }
        int dmafd[req.count] = {0};
        NvBufSurf::NvCommonAllocateParams camparams = {0};
        camparams.memType = NVBUF_MEM_SURFACE_ARRAY;
        camparams.width = front ? FRONT_WIDTH : VIDEO_WIDTH;
        camparams.height = front ? FRONT_HEIGHT : VIDEO_HEIGHT;
        camparams.layout = NVBUF_LAYOUT_PITCH;
        camparams.colorFormat = NVBUF_COLOR_FORMAT_UYVY;
        camparams.memtag = NvBufSurfaceTag_CAMERA;
        if (NvBufSurf::NvAllocate(&camparams, req.count, dmafd))
        {
            perror("NvBufSurf::NvAllocate");
            return ;
        }
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;
        for (int i = 0; i < req.count; i++)
        {
            buf.index = i;
            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0)
            {
                perror("Querying Buffer");
                close(fd);
                return ;
            }
            buf.m.fd = dmafd[i];
            m_videodma[fd].push_back(dmafd[i]);
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0)
            {
                perror("Querying Buffer");
                close(fd);
                return ;
            }
        }
        ioctl(fd, VIDIOC_STREAMON, &buf.type);
    }

    std::vector<int> m_videofd;
    std::map<int, std::vector<int>> m_videodma;
    std::vector<std::thread> m_process;
    std::function<void(int, int, uint64_t)> m_callback = nullptr;
    bool m_quit = false;
};