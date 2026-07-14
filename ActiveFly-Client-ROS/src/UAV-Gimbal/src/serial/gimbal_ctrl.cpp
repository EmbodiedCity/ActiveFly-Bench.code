#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <string>
#include <thread>
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>

// ROS 相关头文件
#include <ros/ros.h>
#include <std_msgs/String.h>
#include "gimbal/GimbalCmd.h"    // msg
#include "gimbal/GimbalState.h"  // msg

#define PTR_OK(p) ((p) != nullptr)

#pragma pack(1)
// 上位机发送数据包
typedef struct
{
    uint8_t sync[2];  // 同步头 0xA9 0x5B
    struct
    {
        uint8_t trig : 3;   // 命令触发计数
        uint8_t value : 5;  // 命令码 0- 未定义；1- 陀螺仪校准；2- 启动云台；3- 停止云台；4- 手动控制；5- 指点平移
    } cmd;
    struct
    {
        uint8_t : 3;
        int8_t f_sens : 5;  // FPV 跟随灵敏度
    } aux;
    struct
    {
        uint8_t : 3;
        uint8_t go_zero : 1;  // 回中触发 变化触发
        uint8_t wk_mode : 2;  // 工作模式 0- 跟随；1- 锁定；2- FPV
        uint8_t op_type : 2;  // 控制模式 0- 角度控制；1- 比例角速度控制；2- 真实角速度控制
        int16_t op_value;     // 控制量，锁定 + 角度控制模式下，姿态角 = op_value * 0.01(deg)
    } gbc[3];                 // 0-滚转，1-俯仰，2-偏航
    struct
    {
        uint8_t : 7;
        uint8_t valid : 1;  // 载机惯导数据有效标志 0- 无效；1- 有效
        int16_t angle[3];   // 载机姿态角 0- 滚转，[-18000,18000)； 1- 俯仰， [-9000,9000]； 2- 偏航，[-18000,18000)
        int16_t accel[3];   // 载机加速度 0- 北向（北向为正）；1- 东向（东向为正） ；2- 天向（天向为正)
    } uav;
    struct
    {
        uint32_t vert_fovlx : 7;   // 相机 1x 垂直视场角
        uint32_t zoom_value : 24;  // 相机倍率
        uint32_t reserved : 1;
        float target_angle[2];  // 目标偏移角度
    } cam;
    uint8_t crc[2];  // CRC16 校验
} Gcu2GbcPkt_t;

// 云台返回数据包
typedef struct
{
    uint8_t sync[2];       // 同步头 0xB5 0x9A
    uint8_t fw_ver;        // 固件版本号
    uint8_t hw_err;        // 硬件错误码
    uint8_t inv_fag : 1;   // 吊装/立装标志 0- 吊装；1- 立装
    uint8_t gbc_stat : 3;  // 云台状态 0- 未定义；1- 初始化中；2- 云台停止；3- 云台保护；4- 手动控制；5- 指点平移
    uint8_t tca_fag : 1;   // 温控就绪标志 0- 未就绪
    uint8_t : 3;
    struct
    {
        uint8_t stat : 3;   // 命令执行状态 0- 执行中；1- 执行成功；2- 执行失败
        uint8_t value : 5;  // 命令码回报
    } cmd;
    int16_t cam_rate[3];   // 相机姿态角速度 0- 俯仰；1- 滚转；2- 偏航
    int16_t cam_angle[3];  // 相机姿态角 0- 滚转；1- 俯仰；2- 偏航
    int16_t mtr_angle[3];  // 电机相对角 0- 俯仰；1- 滚转；2- 偏航
    uint8_t crc[2];        // CRC16 校验
} Gbc2GcuPkt_t;
#pragma pack()


// 云台 CRC16 计算
uint16_t calculateCrc16(const uint8_t *ptr, uint8_t len)
{
    uint16_t crc;
    uint8_t da;
    uint16_t crc_ta[16] = {0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7, 0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef};
    crc = 0;
    while (len-- != 0)
    {
        da = crc >> 12;
        crc <<= 4;
        crc ^= crc_ta[da ^ (*ptr >> 4)];
        da = crc >> 12;
        crc <<= 4;
        crc ^= crc_ta[da ^ (*ptr & 0x0F)];
        ptr++;
    }
    return (crc);
}

// 串口通信类
class SerialPort {
  public:
    // 默认构造
    SerialPort() = default;

    explicit SerialPort(const std::string &tty_path)
    {
        openSerial(tty_path);
    }

    ~SerialPort()
    {
        closeSerial();
    }

    // 配置为 115200-8N1 的 raw 模式
    bool openSerial(const std::string &tty_path)
    {
        closeSerial();

        fd_ = ::open(tty_path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0)
        {
            return false;
        }

        // 切换为阻塞模式，读超时统一由 select 控制
        int flags = fcntl(fd_, F_GETFL, 0);
        if (flags >= 0)
        {
            (void)fcntl(fd_, F_SETFL, flags & (~O_NONBLOCK));
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        if (tcgetattr(fd_, &tty) != 0)
        {
            closeSerial();
            return false;
        }

        // 保存打开前的原始配置
        original_termios_ = tty;

        cfmakeraw(&tty);

        // 云台串口参数：115200、8 数据位、无校验、1 停止位
        cfsetispeed(&tty, B115200);
        cfsetospeed(&tty, B115200);
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cflag |= CLOCAL | CREAD;

        // 读超时由 select 管理
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;

        if (tcsetattr(fd_, TCSANOW, &tty) != 0)
        {
            closeSerial();
            return false;
        }

        // 清空历史收发缓存
        tcflush(fd_, TCIOFLUSH);
        is_open_ = true;
        return true;
    }

    // 关闭串口并恢复原始 termios 配置
    void closeSerial()
    {
        if (fd_ >= 0)
        {
            if (is_open_)
            {
                (void)tcsetattr(fd_, TCSADRAIN, &original_termios_);
            }
            ::close(fd_);
            fd_ = -1;
        }
        is_open_ = false;
    }

    bool isOpen() const
    {
        return is_open_;
    }

    // 阻塞发送全部数据，返回实际发送字节数
    ssize_t writeData(const char *data, size_t len)
    {
        if (!is_open_ || data == nullptr || len == 0)
        {
            return -1;
        }

        size_t total = 0;
        while (total < len)
        {
            ssize_t n = ::write(fd_, data + total, len - total);
            if (n < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                return -1;
            }
            total += static_cast<size_t>(n);
        }

        return static_cast<ssize_t>(total);
    }

    // 发送 Gcu2GbcPkt_t
    ssize_t writeGcu2GbcPkt_t(const Gcu2GbcPkt_t &pkt)
    {
        return writeData(reinterpret_cast<const char *>(&pkt), sizeof(Gcu2GbcPkt_t));
    }

    // timeout_ms < 0: 永久阻塞；timeout_ms == 0: 立即返回；timeout_ms > 0: 毫秒超时
    ssize_t readData(char *buffer, size_t len, int timeout_ms = -1)
    {
        if (!is_open_ || buffer == nullptr || len == 0)
        {
            return -1;
        }

        for (;;)
        {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(fd_, &readfds);

            struct timeval tv;
            struct timeval *tv_ptr = nullptr;
            if (timeout_ms >= 0)
            {
                tv.tv_sec = timeout_ms / 1000;
                tv.tv_usec = (timeout_ms % 1000) * 1000;
                tv_ptr = &tv;
            }

            int ready = select(fd_ + 1, &readfds, nullptr, nullptr, tv_ptr);
            if (ready < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                return -1;
            }

            // 超时且没有数据时返回 0
            if (ready == 0)
            {
                return 0;
            }

            for (;;)
            {
                ssize_t n = ::read(fd_, buffer, len);
                if (n < 0 && errno == EINTR)
                {
                    continue;
                }
                return n;
            }
        }
    }

    // 固定读取一个 Gbc2GcuPkt_t 数据包
    ssize_t readGbc2GcuPkt_t(Gbc2GcuPkt_t *pkt, int first_timeout_ms = 50, int inter_byte_timeout_ms = 3)
    {
        if (!is_open_ || pkt == nullptr)
        {
            return -1;
        }

        constexpr uint8_t kSync0 = 0xB5;
        constexpr uint8_t kSync1 = 0x9A;
        const size_t need = sizeof(Gbc2GcuPkt_t);
        uint8_t *raw = reinterpret_cast<uint8_t *>(pkt);

        // 同步头
        bool got_sync0 = false;
        for (;;)
        {
            uint8_t ch = 0;
            const int timeout_ms = got_sync0 ? inter_byte_timeout_ms : first_timeout_ms;
            const ssize_t n = readData(reinterpret_cast<char *>(&ch), 1, timeout_ms);
            if (n <= 0)
            {
                return n;
            }

            if (!got_sync0)
            {
                if (ch == kSync0)
                {
                    raw[0] = ch;
                    got_sync0 = true;
                }
                continue;
            }

            if (ch == kSync1)
            {
                raw[1] = ch;
                break;
            }

            // 若当前字节仍等于首同步字节，则将其作为新的起点
            if (ch == kSync0)
            {
                raw[0] = ch;
                got_sync0 = true;
            }
            else
            {
                got_sync0 = false;
            }
        }

        size_t total = 2;
        while (total < need)
        {
            const ssize_t n = readData(reinterpret_cast<char *>(raw + total), need - total, inter_byte_timeout_ms);
            if (n <= 0)
            {
                return n;
            }
            total += static_cast<size_t>(n);
        }

        const uint16_t crc_calc = calculateCrc16(raw, static_cast<uint8_t>(need - 2));
        const uint16_t crc_recv = (static_cast<uint16_t>(raw[need - 2]) << 8) | raw[need - 1];
        if (crc_calc != crc_recv)
        {
            return -2;
        }

        return static_cast<ssize_t>(need);
    }

    int flushInput()
    {
        if (!is_open_)
        {
            return -1;
        }
        return tcflush(fd_, TCIFLUSH);
    }

  private:
    int fd_ = -1;                      // 串口文件描述符，-1 表示未打开
    bool is_open_ = false;             // 串口逻辑打开状态
    struct termios original_termios_;  // 打开前的原始配置，用于关闭时恢复
};


// 云台节点类
class GimbalBridgeNode {
  public:
    GimbalBridgeNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : stop_token_(false)
    {
        pnh.param<std::string>("serial_path", serial_path_, serial_path_);
        pnh.param("debug_tx_packets", debug_tx_packets_, debug_tx_packets_);
        pnh.param("debug_rx_packets", debug_rx_packets_, debug_rx_packets_);
        pnh.param("locked_roll_deg", locked_roll_deg_, locked_roll_deg_);
        pnh.param("locked_pitch_deg", locked_pitch_deg_, locked_pitch_deg_);
        pnh.param("locked_yaw_deg", locked_yaw_deg_, locked_yaw_deg_);
        pnh.param("pitch_ctrl_min_deg", pitch_ctrl_min_deg_, pitch_ctrl_min_deg_);
        pnh.param("pitch_ctrl_max_deg", pitch_ctrl_max_deg_, pitch_ctrl_max_deg_);
        pnh.param("check_mount_mode", check_mount_mode_, check_mount_mode_);
        pnh.param("expected_inv_flag", expected_inv_flag_, expected_inv_flag_);
        pnh.param("yaw_wk_mode", yaw_wk_mode_, yaw_wk_mode_);
        pnh.param("fpv_follow_sens", fpv_follow_sens_, fpv_follow_sens_);
        pnh.param("rx_first_timeout_ms", rx_first_timeout_ms_, rx_first_timeout_ms_);
        pnh.param("io_sleep_ms", io_sleep_ms_, io_sleep_ms_);

        const double raw_pitch_ctrl_min_deg = pitch_ctrl_min_deg_;
        const double raw_pitch_ctrl_max_deg = pitch_ctrl_max_deg_;

        locked_roll_deg_ = clampValue(locked_roll_deg_, -180.0, 180.0);
        pitch_ctrl_min_deg_ = clampValue(pitch_ctrl_min_deg_, -90.0, 90.0);
        pitch_ctrl_max_deg_ = clampValue(pitch_ctrl_max_deg_, -90.0, 90.0);
        if (pitch_ctrl_min_deg_ > pitch_ctrl_max_deg_)
        {
            ROS_WARN_STREAM("pitch_ctrl_min_deg(" << pitch_ctrl_min_deg_ << ") > pitch_ctrl_max_deg(" << pitch_ctrl_max_deg_ << "), swap them");
            const double tmp = pitch_ctrl_min_deg_;
            pitch_ctrl_min_deg_ = pitch_ctrl_max_deg_;
            pitch_ctrl_max_deg_ = tmp;
        }

        if (pitch_ctrl_min_deg_ != raw_pitch_ctrl_min_deg || pitch_ctrl_max_deg_ != raw_pitch_ctrl_max_deg)
        {
            ROS_WARN_STREAM("pitch control range adjusted to [" << pitch_ctrl_min_deg_ << ", " << pitch_ctrl_max_deg_ << "] deg");
        }

        locked_pitch_deg_ = clampValue(locked_pitch_deg_, pitch_ctrl_min_deg_, pitch_ctrl_max_deg_);
        locked_yaw_deg_ = clampValue(locked_yaw_deg_, -180.0, 180.0);
        locked_roll_cdeg_ = toCentiDeg(locked_roll_deg_);
        locked_pitch_cdeg_ = toCentiDeg(locked_pitch_deg_);
        locked_yaw_cdeg_ = toCentiDeg(locked_yaw_deg_);
        pitch_ctrl_min_cdeg_ = toCentiDeg(pitch_ctrl_min_deg_);
        pitch_ctrl_max_cdeg_ = toCentiDeg(pitch_ctrl_max_deg_);
        target_pitch_cdeg_ = static_cast<int16_t>(clampInt(static_cast<int>(locked_pitch_cdeg_), pitch_ctrl_min_cdeg_, pitch_ctrl_max_cdeg_));

        const int raw_yaw_wk_mode = yaw_wk_mode_;
        const int raw_fpv_follow_sens = fpv_follow_sens_;
        const int raw_rx_first_timeout_ms = rx_first_timeout_ms_;
        const int raw_io_sleep_ms = io_sleep_ms_;

        yaw_wk_mode_ = clampInt(yaw_wk_mode_, 0, 2);
        fpv_follow_sens_ = clampInt(fpv_follow_sens_, -16, 15);
        rx_first_timeout_ms_ = clampInt(rx_first_timeout_ms_, 1, 50);
        io_sleep_ms_ = clampInt(io_sleep_ms_, 0, 20);

        if (yaw_wk_mode_ != raw_yaw_wk_mode)
        {
            ROS_WARN_STREAM("invalid yaw_wk_mode=" << raw_yaw_wk_mode << ", clamp to " << yaw_wk_mode_);
        }
        if (fpv_follow_sens_ != raw_fpv_follow_sens)
        {
            ROS_WARN_STREAM("invalid fpv_follow_sens=" << raw_fpv_follow_sens << ", clamp to " << fpv_follow_sens_);
        }
        if (rx_first_timeout_ms_ != raw_rx_first_timeout_ms)
        {
            ROS_WARN_STREAM("invalid rx_first_timeout_ms=" << raw_rx_first_timeout_ms << ", clamp to " << rx_first_timeout_ms_);
        }
        if (io_sleep_ms_ != raw_io_sleep_ms)
        {
            ROS_WARN_STREAM("invalid io_sleep_ms=" << raw_io_sleep_ms << ", clamp to " << io_sleep_ms_);
        }

        if (expected_inv_flag_ != 0 && expected_inv_flag_ != 1)
        {
            ROS_WARN_STREAM("invalid expected_inv_flag=" << expected_inv_flag_ << ", fallback to 0(hanging/inverted)");
            expected_inv_flag_ = 0;
        }

        // 错误话题
        fault_pub_ = nh.advertise<std_msgs::String>("/gimbal/fault", 10);
        // 状态话题
        state_pub_ = nh.advertise<gimbal::GimbalState>("/gimbal/state", 10);
        // 控制话题
        cmd_sub_ = nh.subscribe("/gimbal/cmd", 10, &GimbalBridgeNode::onCtrlMsg, this);

        ROS_INFO_STREAM("gimbal_ctrl config: serial_path=" << serial_path_
                        << ", debug_tx_packets=" << (debug_tx_packets_ ? "true" : "false")
                        << ", debug_rx_packets=" << (debug_rx_packets_ ? "true" : "false")
                        << ", locked_roll_deg=" << locked_roll_deg_
                        << ", locked_pitch_deg=" << locked_pitch_deg_
                        << ", locked_yaw_deg=" << locked_yaw_deg_
                        << ", pitch_ctrl_range_deg=[" << pitch_ctrl_min_deg_ << "," << pitch_ctrl_max_deg_ << "]"
                        << ", yaw_wk_mode=" << yaw_wk_mode_
                        << ", fpv_follow_sens=" << fpv_follow_sens_
                        << ", rx_first_timeout_ms=" << rx_first_timeout_ms_
                        << ", io_sleep_ms=" << io_sleep_ms_
                        << ", check_mount_mode=" << (check_mount_mode_ ? "true" : "false")
                        << ", expected_inv_flag=" << expected_inv_flag_);

        // 初始化串口
        if (serial_.openSerial(serial_path_))
        {
            // 创建通信线程
            io_thread_ = std::thread(&GimbalBridgeNode::ioLoop, this);
        }
        else
        {
            ROS_FATAL_STREAM(serial_path_ << " open failed!!!");
            /// @todo 添加错误消息发布、恢复策略等处理
        }
    }

    ~GimbalBridgeNode()
    {
        stop_token_ = true;
        if (io_thread_.joinable())
        {
            io_thread_.join();
        }
    }

    /// @todo 需要改为手动启动，避免类构建函数内启动
    void startComm()
    {
    }

  private:
    /// 订阅控制消息回调，后续可替换为更合适的 ROS 消息格式
    void onCtrlMsg(const gimbal::GimbalCmd::ConstPtr &msg)
    {
        ROS_DEBUG_STREAM_THROTTLE(1.0,
                                  "recv gimbal cmd: ctrl_sw=" << static_cast<int>(msg->ctrl_sw)
                                                               << ", cmd_value=" << static_cast<int>(msg->cmd_value)
                                                               << ", [p,r,y]=[" << msg->op_value_p << ","
                                                               << msg->op_value_r << "," << msg->op_value_y << "]");
        {
            std::lock_guard<std::mutex> lock(tx_mtx_);
            if (modifyTxGbc(&tx_pkt_, msg) == 0)
            {
                refreshTxCrc(&tx_pkt_);
            }
            else
            {
                ROS_ERROR_STREAM("modify tx_gbc error");
                /// @todo 添加修改错误上报
            }
        }
    }

    // 云台回包并发布状态消息
    int publishReply(const Gbc2GcuPkt_t &data)
    {
        gimbal::GimbalState gim_state;
        gim_state.inv_fag = data.inv_fag;
        gim_state.gbc_stat = data.gbc_stat;
        gim_state.tca_fag = data.tca_fag;
        gim_state.cmd_stat = data.cmd.stat;
        gim_state.cmd_value = data.cmd.value;

        // 姿态角速度
        gim_state.cam_rate_p = data.cam_rate[0];
        gim_state.cam_rate_r = data.cam_rate[1];
        gim_state.cam_rate_y = data.cam_rate[2];

        // 相机姿态角
        gim_state.cam_angle_p = data.cam_angle[1];
        gim_state.cam_angle_r = data.cam_angle[0];
        gim_state.cam_angle_y = data.cam_angle[2];

        // 电机相对角
        gim_state.mtr_angle_p = data.mtr_angle[0];
        gim_state.mtr_angle_r = data.mtr_angle[1];
        gim_state.mtr_angle_y = data.mtr_angle[2];

        state_pub_.publish(gim_state);
        return 0;
    }

    /// @todo 错误信息上报待完善
    int publishFault(int err_code, const std::string &err_info)
    {
        (void)err_code;
        (void)err_info;
        return 0;
    }

    // 云台通信线程
    void ioLoop()
    {
        int num = 0;
        int rx_timeout_cnt = 0;  // 读取超时
        int rx_crc_err_cnt = 0;  // crc校验不通过
        int rx_io_err_cnt = 0;   // 其他
        bool has_last_rx = false;
        bool switch_to_manual_after_first_reply = true;
        Gbc2GcuPkt_t last_rx_pkt{};

        // 初始化发送结构体
        initTxPkt(&tx_pkt_);
        // 上电后先持续发送启动命令，收到首个有效回包后再切手动控制。
        modifyTxCmd(&tx_pkt_, 2);
        refreshTxCrc(&tx_pkt_);

        ROS_INFO_STREAM("gimbal comm start on " << serial_path_
                        << ", startup cmd=2 then switch to cmd=4 after first valid reply");

        // 持续收发数据
        while (!stop_token_ && ros::ok())
        {
            int r_ret = 0;
            Gbc2GcuPkt_t rx_pkt{};
            {  // 在锁内完成一次完整的收发流程
                std::lock_guard<std::mutex> lock(tx_mtx_);
                enforceAxisModesAndTargets(&tx_pkt_);
                refreshTxCrc(&tx_pkt_);

                if (debug_tx_packets_)
                {
                    outTxPkt(tx_pkt_);
                }
                if (serial_.writeGcu2GbcPkt_t(tx_pkt_) == sizeof(Gcu2GbcPkt_t))
                {
                    // 首字节超时不能设为 0，否则会立即返回
                    r_ret = serial_.readGbc2GcuPkt_t(&rx_pkt, rx_first_timeout_ms_, 3);
                }
                else
                {
                    ROS_ERROR_STREAM_THROTTLE(2.0, "gimbal serial write error");
                    r_ret = -1;
                }
            }

            if (r_ret == static_cast<int>(sizeof(Gbc2GcuPkt_t)))
            {
                has_last_rx = true;
                last_rx_pkt = rx_pkt;
                checkMountMode(rx_pkt);
                if (debug_rx_packets_)
                {
                    outRxPkt(rx_pkt);
                }

                if (switch_to_manual_after_first_reply)
                {
                    std::lock_guard<std::mutex> lock(tx_mtx_);
                    if (tx_pkt_.cmd.value == 2)
                    {
                        modifyTxCmd(&tx_pkt_, 4);
                        refreshTxCrc(&tx_pkt_);
                        ROS_INFO("received first valid gimbal reply, switch to manual control cmd=4");
                    }
                    switch_to_manual_after_first_reply = false;
                }

                publishReply(rx_pkt);
            }
            else if (r_ret == 0)
            {
                rx_timeout_cnt++;
                if (!has_last_rx)
                {
                    ROS_WARN_STREAM_THROTTLE(15.0, "gimbal read timeout, no valid reply yet from " << serial_path_);
                }
            }
            else if (r_ret == -2)
            {
                rx_crc_err_cnt++;
                ROS_WARN_STREAM_THROTTLE(15.0, "gimbal crc check failed");
            }
            else
            {
                rx_io_err_cnt++;
                ROS_WARN_STREAM_THROTTLE(15.0, "gimbal serial io error");
            }

            num++;
            if (num > 2000)
            {
                if (rx_timeout_cnt > 0 || rx_crc_err_cnt > 0 || rx_io_err_cnt > 0 || !has_last_rx)
                {
                    ROS_DEBUG_STREAM("gimbal rx stat: timeout=" << rx_timeout_cnt
                                     << ", crc_err=" << rx_crc_err_cnt
                                     << ", io_err=" << rx_io_err_cnt
                                     << ", has_valid_reply=" << (has_last_rx ? "true" : "false"));
                }

                num = 0;

                /// @todo 串口通讯的状态进行上报、统计等状态信息的交互完善
                rx_timeout_cnt = 0;
                rx_crc_err_cnt = 0;
                rx_io_err_cnt = 0;
            }

            // 预留可调 sleep，便于在时延和 CPU 占用之间折中。
            std::this_thread::sleep_for(std::chrono::milliseconds(io_sleep_ms_));
        }
    }

    void outTxPkt(const Gcu2GbcPkt_t &tx_pkt)
    {
        const uint8_t *raw = reinterpret_cast<const uint8_t *>(&tx_pkt);
        const size_t len = sizeof(Gcu2GbcPkt_t);
        const uint16_t crc_calc = calculateCrc16(raw, static_cast<uint8_t>(len - 2));
        const uint16_t crc_send = (static_cast<uint16_t>(tx_pkt.crc[0]) << 8) | tx_pkt.crc[1];

        auto cmdName = [](uint8_t v) -> const char * {
            switch (v)
            {
            case 1:
                return "陀螺仪校准";
            case 2:
                return "启动云台";
            case 3:
                return "停止云台";
            case 4:
                return "手动控制";
            case 5:
                return "指点平移";
            default:
                return "未定义";
            }
        };
        auto wkModeName = [](uint8_t v) -> const char * {
            switch (v)
            {
            case 0:
                return "跟随";
            case 1:
                return "锁定";
            case 2:
                return "FPV";
            default:
                return "保留";
            }
        };
        auto opTypeName = [](uint8_t v) -> const char * {
            switch (v)
            {
            case 0:
                return "角度控制";
            case 1:
                return "比例角速度";
            case 2:
                return "真实角速度";
            default:
                return "保留";
            }
        };
        const char *axis[3] = {"roll", "pitch", "yaw"};

        std::cout << "\n========== TX ==========\n";
        std::cout << "[TX] len=" << len
                  << " sync=0x" << std::hex << static_cast<int>(tx_pkt.sync[0])
                  << " 0x" << static_cast<int>(tx_pkt.sync[1]) << std::dec
                  << " crc=" << ((crc_send == crc_calc) ? "OK" : "BAD") << "\n";

        std::cout << "cmd: trig=" << static_cast<int>(tx_pkt.cmd.trig)
                  << " value=" << static_cast<int>(tx_pkt.cmd.value)
                  << "(" << cmdName(tx_pkt.cmd.value) << ")\n";

        for (int i = 0; i < 3; ++i)
        {
            std::cout << axis[i]
                      << ": go_zero=" << static_cast<int>(tx_pkt.gbc[i].go_zero)
                      << " wk_mode=" << static_cast<int>(tx_pkt.gbc[i].wk_mode) << "(" << wkModeName(tx_pkt.gbc[i].wk_mode) << ")"
                      << " op_type=" << static_cast<int>(tx_pkt.gbc[i].op_type) << "(" << opTypeName(tx_pkt.gbc[i].op_type) << ")"
                      << " op_value=" << tx_pkt.gbc[i].op_value << "\n";
        }

        std::cout << "uav.valid=" << static_cast<int>(tx_pkt.uav.valid)
                  << " angle[r,p,y]=[" << tx_pkt.uav.angle[0] << ","
                  << tx_pkt.uav.angle[1] << "," << tx_pkt.uav.angle[2] << "]\n";
    }

    void outRxPkt(const Gbc2GcuPkt_t &rx_pkt)
    {
        const uint8_t *raw = reinterpret_cast<const uint8_t *>(&rx_pkt);
        const size_t len = sizeof(Gbc2GcuPkt_t);
        const uint16_t crc_calc = calculateCrc16(raw, static_cast<uint8_t>(len - 2));
        const uint16_t crc_recv = (static_cast<uint16_t>(rx_pkt.crc[0]) << 8) | rx_pkt.crc[1];

        auto gbcStatName = [](uint8_t v) -> const char * {
            switch (v)
            {
            case 1:
                return "初始化中";
            case 2:
                return "云台停止";
            case 3:
                return "云台保护";
            case 4:
                return "手动控制";
            case 5:
                return "指点平移";
            default:
                return "未定义";
            }
        };
        auto cmdStatName = [](uint8_t v) -> const char * {
            switch (v)
            {
            case 0:
                return "执行中";
            case 1:
                return "成功";
            case 2:
                return "失败";
            default:
                return "保留";
            }
        };

        std::cout << "\n========== RX ==========\n";
        std::cout << "[RX] len=" << len
                  << " sync=0x" << std::hex << static_cast<int>(rx_pkt.sync[0])
                  << " 0x" << static_cast<int>(rx_pkt.sync[1]) << std::dec
                  << " crc=" << ((crc_recv == crc_calc) ? "OK" : "BAD") << "\n";

        std::cout << "fw=" << static_cast<int>(rx_pkt.fw_ver)
                  << " hw_err=" << static_cast<int>(rx_pkt.hw_err)
                  << " inv=" << static_cast<int>(rx_pkt.inv_fag) << (rx_pkt.inv_fag ? "(立装)" : "(吊装)")
                  << " tca=" << static_cast<int>(rx_pkt.tca_fag) << "\n";

        std::cout << "gbc_stat=" << static_cast<int>(rx_pkt.gbc_stat) << "(" << gbcStatName(rx_pkt.gbc_stat) << ")"
                  << " cmd_stat=" << static_cast<int>(rx_pkt.cmd.stat) << "(" << cmdStatName(rx_pkt.cmd.stat) << ")"
                  << " cmd_value=" << static_cast<int>(rx_pkt.cmd.value) << "\n";

        // cam_rate[俯仰,滚转,偏航]；cam_angle[滚转,俯仰,偏航]；mtr_angle[俯仰,滚转,偏航]
        // 俯仰 p ,滚转 r,偏航 y
        std::cout << std::fixed << std::setprecision(2) << "角速度(deg/s) [p,r,y]=[" << std::setw(7) << rx_pkt.cam_rate[0] * 0.01f << "," << std::setw(7) << rx_pkt.cam_rate[1] * 0.01f << "," << std::setw(7) << rx_pkt.cam_rate[2] * 0.01f << "]" << std::defaultfloat << "\n";
        std::cout << std::fixed << std::setprecision(2) << "姿态角(deg)   [p,r,y]=[" << std::setw(7) << rx_pkt.cam_angle[1] * 0.01f << "," << std::setw(7) << rx_pkt.cam_angle[0] * 0.01f << "," << std::setw(7) << rx_pkt.cam_angle[2] * 0.01f << "]" << std::defaultfloat << "\n";
        std::cout << std::fixed << std::setprecision(2) << "相对角(deg)   [p,r,y]=[" << std::setw(7) << rx_pkt.mtr_angle[0] * 0.01f << "," << std::setw(7) << rx_pkt.mtr_angle[1] * 0.01f << "," << std::setw(7) << rx_pkt.mtr_angle[2] * 0.01f << "]" << std::defaultfloat << "\n";
    }


    /// @warning 发送包字段修改函数默认不加锁，调用方负责同步

    // 刷新发送包的 CRC16 校验码
    void refreshTxCrc(Gcu2GbcPkt_t *tx_pkt)
    {
        if (!PTR_OK(tx_pkt))
        {
            return;
        }
        // 计算并写回 CRC
        uint16_t crc = calculateCrc16(reinterpret_cast<const uint8_t *>(tx_pkt), sizeof(Gcu2GbcPkt_t) - 2);
        tx_pkt->crc[0] = (crc >> 8) & 0xFF;
        tx_pkt->crc[1] = crc & 0xFF;
    }

    // 初始化发送结构体
    void initTxPkt(Gcu2GbcPkt_t *tx_pkt)
    {
        if (!PTR_OK(tx_pkt))
        {
            return;
        }
        // 清零整包并写入固定同步头
        memset(tx_pkt, 0, sizeof(Gcu2GbcPkt_t));
        tx_pkt->sync[0] = 0xA9;
        tx_pkt->sync[1] = 0x5B;

        // 默认不触发额外命令
        tx_pkt->cmd.trig = 0;
        tx_pkt->cmd.value = 0;

        // 默认上电姿态：roll 锁定 0 度，yaw 锁定 -90 度，pitch 为可控初值。
        enforceAxisModesAndTargets(tx_pkt);

        tx_pkt->uav.valid = 0;  // 默认载机姿态数据无效

        refreshTxCrc(tx_pkt);
    }

    // 修改发送包命令码
    int modifyTxCmd(Gcu2GbcPkt_t *tx_pkt, int cmd)
    {
        if (!PTR_OK(tx_pkt))
        {
            return -1;
        }
        tx_pkt->cmd.value = cmd;
        return 0;
    }

    // 修改下发包数据 
    int modifyTxGbc(Gcu2GbcPkt_t *tx_pkt, const gimbal::GimbalCmd::ConstPtr &msg)
    {
        if (!PTR_OK(tx_pkt))
        {
            return -1;
        }

        // roll/yaw 轴锁定固定角，pitch 轴锁定角模式下实时可控。
        enforceAxisModesAndTargets(tx_pkt);

        // 控制选择
        if (msg->ctrl_sw == 1)  // 命令码修改
        {
            if (msg->cmd_value > 5)
            {
                return -1;
            }
            tx_pkt->cmd.value = msg->cmd_value;
        }
        else if (msg->ctrl_sw == 2)  // 回中
        {
            /// @todo 需要考虑是否需要实现
        }
        else if (msg->ctrl_sw == 3)  // 控制姿态角
        {
            tx_pkt->cmd.value = 4;

            const int pitch_cmd_cdeg = clampInt(static_cast<int>(msg->op_value_p), pitch_ctrl_min_cdeg_, pitch_ctrl_max_cdeg_);
            if (pitch_cmd_cdeg != static_cast<int>(msg->op_value_p))
            {
                ROS_WARN_STREAM_THROTTLE(10.0,
                                         "pitch cmd out of range, clamp from " << msg->op_value_p
                                                                             << " to " << pitch_cmd_cdeg
                                                                             << " (0.01deg)");
            }
            target_pitch_cdeg_ = static_cast<int16_t>(pitch_cmd_cdeg);
            tx_pkt->gbc[1].op_value = target_pitch_cdeg_;

            if (msg->op_value_r != 0 || msg->op_value_y != 0)
            {
                ROS_WARN_STREAM_THROTTLE(10.0,
                                         "roll/yaw realtime cmd are ignored in current profile, use op_value_p for pitch control");
            }
        }
        else
        {
            ROS_WARN_STREAM_THROTTLE(10.0, "ctrl_sw not exist");
            /// @todo 回传待完善
        }
        /// @todo 完善指令

        return 0;
    }

    static double clampValue(double value, double min_val, double max_val)
    {
        if (value < min_val)
        {
            return min_val;
        }
        if (value > max_val)
        {
            return max_val;
        }
        return value;
    }

    static int16_t toCentiDeg(double deg)
    {
        return static_cast<int16_t>(std::lround(deg * 100.0));
    }

    static int clampInt(int value, int min_val, int max_val)
    {
        if (value < min_val)
        {
            return min_val;
        }
        if (value > max_val)
        {
            return max_val;
        }
        return value;
    }

    void enforceAxisModesAndTargets(Gcu2GbcPkt_t *tx_pkt)
    {
        if (!PTR_OK(tx_pkt))
        {
            return;
        }

        // roll: 锁定+角度控制，固定在预设角（默认 0 度）。
        tx_pkt->gbc[0].wk_mode = 1;
        tx_pkt->gbc[0].op_type = 0;
        tx_pkt->gbc[0].op_value = locked_roll_cdeg_;

        // pitch: 锁定+角度控制，实时跟踪外部下发目标角（范围可配）。
        tx_pkt->gbc[1].wk_mode = 1;
        tx_pkt->gbc[1].op_type = 0;
        tx_pkt->gbc[1].op_value = target_pitch_cdeg_;

        // yaw: 默认锁定+角度控制（启动即转到预设角，默认 -90 度）。
        tx_pkt->gbc[2].wk_mode = static_cast<uint8_t>(yaw_wk_mode_);
        tx_pkt->gbc[2].op_type = 0;
        tx_pkt->gbc[2].op_value = locked_yaw_cdeg_;

        // FPV 模式下由该参数提升跟随刚性，锁定模式下该字段无副作用。
        tx_pkt->aux.f_sens = static_cast<int8_t>(fpv_follow_sens_);
    }

    void checkMountMode(const Gbc2GcuPkt_t &rx_pkt)
    {
        if (!check_mount_mode_)
        {
            return;
        }

        const uint8_t expected = static_cast<uint8_t>(expected_inv_flag_);
        if (rx_pkt.inv_fag != expected)
        {
            ROS_ERROR_STREAM_THROTTLE(15.0,
                                      "gimbal mount mode mismatch: expected inv_flag=" << static_cast<int>(expected)
                                      << (expected ? "(upright)" : "(hanging/inverted)")
                                      << ", got inv_flag=" << static_cast<int>(rx_pkt.inv_fag)
                                      << (rx_pkt.inv_fag ? "(upright)" : "(hanging/inverted)"));
        }
    }

  private:
    ros::Publisher fault_pub_;
    ros::Publisher state_pub_;
    ros::Subscriber cmd_sub_;

    std::string serial_path_ = "/dev/ttyUSB0";  // 串口地址
    bool debug_tx_packets_ = false;
    bool debug_rx_packets_ = false;
    double locked_roll_deg_ = 0.0;
    double locked_pitch_deg_ = 0.0;
    double locked_yaw_deg_ = 90.0;
    double pitch_ctrl_min_deg_ = -90.0;
    double pitch_ctrl_max_deg_ = 40.0;
    int yaw_wk_mode_ = 0;
    int fpv_follow_sens_ = 15;
    int rx_first_timeout_ms_ = 4;
    int io_sleep_ms_ = 1;
    bool check_mount_mode_ = true;
    int expected_inv_flag_ = 0;
    int16_t locked_roll_cdeg_ = 0;
    int16_t locked_pitch_cdeg_ = 0;
    int16_t locked_yaw_cdeg_ = 9000;
    int16_t pitch_ctrl_min_cdeg_ = -9000;
    int16_t pitch_ctrl_max_cdeg_ = 4000;
    int16_t target_pitch_cdeg_ = 0;
    SerialPort serial_;

    Gcu2GbcPkt_t tx_pkt_;  // 当前发送包缓存
    std::mutex tx_mtx_;

    std::thread io_thread_;  // 与云台进行数据交互的线程
    std::atomic<bool> stop_token_;
};


int main(int argc, char **argv)
{
    // 初始化 ROS 节点 gimbal_ctrl
    ros::init(argc, argv, "gimbal_ctrl");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    GimbalBridgeNode g_opj(nh, pnh);

    ros::spin();
}
