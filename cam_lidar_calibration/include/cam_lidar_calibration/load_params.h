#ifndef load_params_h_
#define load_params_h_

#include <ros/ros.h>

#include <opencv2/core/mat.hpp>

namespace cam_lidar_calibration
{
    struct initial_parameters_t
    {
        bool fisheye_model;
        int lidar_ring_count = 0;
        cv::Size chessboard_pattern_size;
        int square_length;                 // in millimetres
        cv::Size board_dimensions;         // in millimetres
        cv::Point3d cb_translation_error;  // in millimetres
        cv::Mat cameramat, distcoeff;
        std::pair<int, int> image_size;  // in pixels
        std::string camera_topic, camera_info, lidar_topic;
    };

    void loadParams(const ros::NodeHandle& n, initial_parameters_t& i_params);

}  // namespace cam_lidar_calibration

#endif
