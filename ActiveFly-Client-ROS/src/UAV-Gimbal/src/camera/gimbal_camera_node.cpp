#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>

class GimbalCameraNode {
public:
  GimbalCameraNode()
      : nh_(),
        pnh_("~"),
        it_(nh_),
        running_(true),
        latest_frame_seq_(0) {
    loadParams();
    initPublishers();
    initCameraInfoManager();

    capture_thread_ = std::thread(&GimbalCameraNode::captureLoop, this);
    publish_thread_ = std::thread(&GimbalCameraNode::publishLoop, this);
  }

  ~GimbalCameraNode() {
    stop();
  }

private:
  void loadParams() {
    pnh_.param<std::string>(
        "device", device_,
        "/dev/v4l/by-id/usb-BC-251027-ZW_MF500_camera_01.00.00-video-index0");
    pnh_.param("width", width_, 1920);
    pnh_.param("height", height_, 1080);
    pnh_.param("fps", fps_, 30.0);
    pnh_.param<std::string>("pixel_format", pixel_format_, "MJPG");
    pnh_.param<std::string>("frame_id", frame_id_, "gimbal_camera_optical");
    pnh_.param<std::string>("camera_name", camera_name_, "mf500_gimbal");
    pnh_.param<std::string>("camera_info_url", camera_info_url_, "");
    pnh_.param("publish_compat_topic", publish_compat_topic_, true);
    pnh_.param<std::string>("compat_topic_name", compat_topic_name_, "/cd_output_img_topic");
    pnh_.param("reconnect_interval_sec", reconnect_interval_sec_, 1.0);
  }

  void initPublishers() {
    camera_pub_ = it_.advertiseCamera("/gimbal/image_raw", 1);
    if (publish_compat_topic_) {
      compat_pub_ = it_.advertise(compat_topic_name_, 1);
    }
  }

  void initCameraInfoManager() {
    if (camera_info_url_.empty()) {
      ROS_WARN_STREAM(
          "camera_info_url is empty. /gimbal/camera_info will be published as "
          "uncalibrated until a calibration file is provided.");
      return;
    }

    ros::NodeHandle camera_nh("/gimbal");
    cinfo_manager_.reset(
        new camera_info_manager::CameraInfoManager(camera_nh, camera_name_));

    if (!cinfo_manager_->validateURL(camera_info_url_)) {
      ROS_WARN_STREAM("camera_info_url is invalid: " << camera_info_url_);
      cinfo_manager_.reset();
      return;
    }

    if (!cinfo_manager_->loadCameraInfo(camera_info_url_)) {
      ROS_WARN_STREAM("Failed to load camera calibration from "
                      << camera_info_url_);
      cinfo_manager_.reset();
      return;
    }

    ROS_INFO_STREAM("Loaded camera calibration from " << camera_info_url_);
  }

  void stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
      return;
    }

    frame_cv_.notify_all();

    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    if (publish_thread_.joinable()) {
      publish_thread_.join();
    }

    if (capture_.isOpened()) {
      capture_.release();
    }
  }

  void captureLoop() {
    while (running_ && ros::ok()) {
      if (!capture_.isOpened() && !openCapture()) {
        ros::WallDuration(reconnect_interval_sec_).sleep();
        continue;
      }

      cv::Mat frame;
      if (!capture_.read(frame) || frame.empty()) {
        ROS_WARN_STREAM_THROTTLE(
            2.0, "Failed to read a frame from "
                     << active_device_path_ << " via " << active_backend_name_
                     << ". Reconnecting...");
        capture_.release();
        active_device_path_.clear();
        active_backend_name_.clear();
        ros::WallDuration(reconnect_interval_sec_).sleep();
        continue;
      }

      cv::Mat bgr_frame = convertToBgr(frame);
      if (bgr_frame.empty()) {
        ROS_WARN_THROTTLE(2.0, "Received an unsupported frame format from the camera.");
        continue;
      }

      const ros::Time stamp = ros::Time::now();
      {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = bgr_frame;
        latest_stamp_ = stamp;
        ++latest_frame_seq_;
      }
      frame_cv_.notify_one();
    }
  }

  void publishLoop() {
    uint64_t published_seq = 0;

    while (running_ && ros::ok()) {
      cv::Mat frame;
      ros::Time stamp;
      uint64_t frame_seq = 0;

      {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        frame_cv_.wait_for(lock, std::chrono::milliseconds(200), [&]() {
          return !running_ || latest_frame_seq_ > published_seq;
        });

        if (!running_ || !ros::ok()) {
          break;
        }

        if (latest_frame_seq_ <= published_seq || latest_frame_.empty()) {
          continue;
        }

        frame = latest_frame_.clone();
        stamp = latest_stamp_;
        frame_seq = latest_frame_seq_;
      }

      std_msgs::Header header;
      header.seq = static_cast<uint32_t>(frame_seq);
      header.stamp = stamp;
      header.frame_id = frame_id_;

      sensor_msgs::ImageConstPtr image_msg =
          cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame)
              .toImageMsg();

      sensor_msgs::CameraInfoPtr info_msg(new sensor_msgs::CameraInfo());
      if (cinfo_manager_) {
        *info_msg = cinfo_manager_->getCameraInfo();
      }
      info_msg->header = header;
      if (info_msg->width == 0 || info_msg->height == 0) {
        info_msg->width = static_cast<uint32_t>(frame.cols);
        info_msg->height = static_cast<uint32_t>(frame.rows);
      } else if (info_msg->width != static_cast<uint32_t>(frame.cols) ||
                 info_msg->height != static_cast<uint32_t>(frame.rows)) {
        ROS_WARN_STREAM_THROTTLE(
            5.0, "camera_info resolution " << info_msg->width << "x"
                                           << info_msg->height
                                           << " does not match the live image "
                                           << frame.cols << "x" << frame.rows
                                           << ".");
      }

      camera_pub_.publish(image_msg, info_msg);
      if (publish_compat_topic_) {
        compat_pub_.publish(image_msg);
      }

      published_seq = frame_seq;
    }
  }

  bool openCapture() {
    if (capture_.isOpened()) {
      return true;
    }

    const std::string resolved_device = resolveDevicePath(device_);
    const int requested_fourcc = makeFourcc(pixel_format_);

    std::vector<OpenAttempt> attempts;
    attempts.push_back({resolved_device, cv::CAP_V4L2, false, -1, "CAP_V4L2(path)"});

    int video_index = -1;
    if (parseVideoIndex(resolved_device, &video_index)) {
      attempts.push_back({"", cv::CAP_V4L2, true, video_index, "CAP_V4L2(index)"});
    }

    attempts.push_back({resolved_device, cv::CAP_ANY, false, -1, "CAP_ANY(path)"});
    if (video_index >= 0) {
      attempts.push_back({"", cv::CAP_ANY, true, video_index, "CAP_ANY(index)"});
    }

    for (const OpenAttempt& attempt : attempts) {
      const bool opened = attempt.use_index ? capture_.open(attempt.index, attempt.backend)
                                            : capture_.open(attempt.target, attempt.backend);
      if (!opened) {
        capture_.release();
        continue;
      }

      configureCapture(requested_fourcc);

      active_device_path_ =
          attempt.use_index ? ("/dev/video" + std::to_string(attempt.index))
                            : attempt.target;
      active_backend_name_ = attempt.label;

      const int actual_width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
      const int actual_height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
      const double actual_fps = capture_.get(cv::CAP_PROP_FPS);
      const int actual_fourcc = static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC));

      ROS_INFO_STREAM("Opened gimbal camera on " << active_device_path_
                      << " via " << active_backend_name_
                      << ", requested " << pixel_format_ << " "
                      << width_ << "x" << height_ << " @ " << fps_ << " fps"
                      << ", actual " << fourccToString(actual_fourcc) << " "
                      << actual_width << "x" << actual_height << " @ "
                      << actual_fps << " fps.");
      return true;
    }

    ROS_WARN_STREAM_THROTTLE(2.0, "Failed to open camera device "
                                      << device_ << " (resolved to "
                                      << resolved_device << ").");
    return false;
  }

  int makeFourcc(const std::string& raw_format) const {
    std::string format = raw_format;
    for (char& ch : format) {
      ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }

    if (format.size() != 4) {
      ROS_WARN_STREAM("Invalid pixel_format '" << raw_format
                                              << "', falling back to MJPG.");
      format = "MJPG";
    }

    return cv::VideoWriter::fourcc(format[0], format[1], format[2], format[3]);
  }

  std::string fourccToString(int fourcc) const {
    std::string text(4, ' ');
    text[0] = static_cast<char>(fourcc & 0xff);
    text[1] = static_cast<char>((fourcc >> 8) & 0xff);
    text[2] = static_cast<char>((fourcc >> 16) & 0xff);
    text[3] = static_cast<char>((fourcc >> 24) & 0xff);
    return text;
  }

  void configureCapture(int requested_fourcc) {
    capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    capture_.set(cv::CAP_PROP_CONVERT_RGB, 1.0);
    capture_.set(cv::CAP_PROP_FOURCC, static_cast<double>(requested_fourcc));
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width_));
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height_));
    capture_.set(cv::CAP_PROP_FPS, fps_);
  }

  std::string resolveDevicePath(const std::string& device) const {
    char resolved_path[PATH_MAX];
    if (realpath(device.c_str(), resolved_path) != nullptr) {
      return std::string(resolved_path);
    }
    return device;
  }

  bool parseVideoIndex(const std::string& device_path, int* index) const {
    if (index == nullptr) {
      return false;
    }

    const std::string prefix = "/dev/video";
    if (device_path.compare(0, prefix.size(), prefix) != 0 ||
        device_path.size() == prefix.size()) {
      return false;
    }

    const std::string suffix = device_path.substr(prefix.size());
    for (char ch : suffix) {
      if (!std::isdigit(static_cast<unsigned char>(ch))) {
        return false;
      }
    }

    *index = std::stoi(suffix);
    return true;
  }

  cv::Mat convertToBgr(const cv::Mat& frame) const {
    if (frame.empty()) {
      return cv::Mat();
    }

    if (frame.channels() == 3) {
      return frame.clone();
    }

    cv::Mat converted;
    if (frame.channels() == 1) {
      cv::cvtColor(frame, converted, cv::COLOR_GRAY2BGR);
      return converted;
    }

    if (frame.channels() == 4) {
      cv::cvtColor(frame, converted, cv::COLOR_BGRA2BGR);
      return converted;
    }

    return cv::Mat();
  }

  struct OpenAttempt {
    std::string target;
    int backend;
    bool use_index;
    int index;
    std::string label;
  };

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraPublisher camera_pub_;
  image_transport::Publisher compat_pub_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> cinfo_manager_;

  std::string device_;
  int width_;
  int height_;
  double fps_;
  std::string pixel_format_;
  std::string frame_id_;
  std::string camera_name_;
  std::string camera_info_url_;
  bool publish_compat_topic_;
  std::string compat_topic_name_;
  double reconnect_interval_sec_;
  std::string active_device_path_;
  std::string active_backend_name_;

  std::atomic<bool> running_;
  cv::VideoCapture capture_;
  std::thread capture_thread_;
  std::thread publish_thread_;

  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  cv::Mat latest_frame_;
  ros::Time latest_stamp_;
  uint64_t latest_frame_seq_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "gimbal_camera");
  GimbalCameraNode node;
  ros::waitForShutdown();
  return 0;
}
