#include "cam_lidar_calibration/load_params.h"

namespace cam_lidar_calibration
{
    void loadParams(const ros::NodeHandle& n, initial_parameters_t& i_params)
    {
        int cb_w, cb_h, w, h, e_x, e_y, i_width, i_height;
        n.getParam("camera_topic", i_params.camera_topic);
        n.getParam("camera_info", i_params.camera_info);
        n.getParam("lidar_topic", i_params.lidar_topic);
        n.getParam("chessboard/pattern_size/width", cb_w);
        n.getParam("chessboard/pattern_size/height", cb_h);
        i_params.chessboard_pattern_size = cv::Size(cb_w, cb_h);
        n.getParam("chessboard/square_length", i_params.square_length);
        n.getParam("chessboard/board_dimension/width", w);
        n.getParam("chessboard/board_dimension/height", h);
        i_params.board_dimensions = cv::Size(w, h);
        n.getParam("chessboard/translation_error/x", e_x);
        n.getParam("chessboard/translation_error/y", e_y);
        i_params.cb_translation_error = cv::Point3d(e_x, e_y, 0);
    }
}  // namespace cam_lidar_calibration

