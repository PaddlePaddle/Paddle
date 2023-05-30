#include <iostream>
#include <string>
#include <numeric>

#include "cam_lidar_calibration/point_xyzir.h"
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float64MultiArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

double colmap[50][3]={{0,0,0.5385},{0,0,0.6154},{0,0,0.6923},{0,0,0.7692},{0,0,0.8462},{0,0,0.9231},{0,0,1.0000},{0,0.0769,1.0000},{0,0.1538,1.0000},{0,0.2308,1.0000},{0,0.3846,1.0000},
                      {0,0.4615,1.0000},{0,0.5385,1.0000},{0,0.6154,1.0000},{0,0.6923,1.0000},{0,0.7692,1.0000},{0,0.8462,1.0000},{0,0.9231,1.0000},{0,1.0000,1.0000},{0.0769,1.0000,0.9231},
                      {0.1538,1.0000,0.8462},{0.2308,1.0000,0.7692},{0.3077,1.0000,0.6923},{0.3846,1.0000,0.6154},{0.4615,1.0000,0.5385},{0.5385,1.0000,0.4615},{0.6154,1.0000,0.3846},
                      {0.6923,1.0000,0.3077},{0.7692,1.0000,0.2308},{0.8462,1.0000,0.1538},{0.9231,1.0000,0.0769},{1.0000,1.0000,0},{1.0000,0.9231,0},{1.0000,0.8462,0},{1.0000,0.7692,0},
                      {1.0000,0.6923,0},{1.0000,0.6154,0},{1.0000,0.5385,0},{1.0000,0.4615,0},{1.0000,0.3846,0},{1.0000,0.3077,0},{1.0000,0.2308,0},{1.0000,0.1538,0},{1.0000,0.0769,0},
                      {1.0000,0,0},{0.9231,0,0},{0.8462,0,0},{0.7692,0,0},{0.6923,0,0},{0.6154,0,0}};

struct OptimisationSample
{
    cv::Point3d camera_centre{ 0, 0, 0 };
    cv::Point3d camera_normal{ 0, 0, 0 };
    std::vector<cv::Point3d> camera_corners;
    cv::Point3d lidar_centre{ 0, 0, 0 };
    cv::Point3d lidar_normal{ 0, 0, 0 };
    std::vector<cv::Point3d> lidar_corners;
    std::vector<double> angles_0;
    std::vector<double> angles_1;
    std::vector<double> widths;
    std::vector<double> heights;
    float distance_from_origin;
    double pixeltometre;
    int sample_num;
};

struct Rotation
{
    double roll;  // Rotation optimization variables
    double pitch;
    double yaw;
    operator const std::string() const
    {
        return std::string("{") + "roll:" + std::to_string(roll) + ", pitch:" + std::to_string(pitch) +
                ", yaw:" + std::to_string(yaw) + "}";
    }
    cv::Mat toMat() const
    {
        using cv::Mat_;
        using std::cos;
        using std::sin;

        // Calculate rotation about x axis
        cv::Mat R_x = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll));
        // Calculate rotation about y axis
        cv::Mat R_y = (Mat_<double>(3, 3) << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch));
        // Calculate rotation about z axis
        cv::Mat R_z = (Mat_<double>(3, 3) << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);

        return R_z * R_y * R_x;
    }
};

struct RotationTranslation
{
    Rotation rot;
    double x;
    double y;
    double z;
    operator const std::string() const
    {
        return std::string("{") + "roll:" + std::to_string(rot.roll) + ", pitch:" + std::to_string(rot.pitch) +
                ", yaw:" + std::to_string(rot.yaw) + ", x:" + std::to_string(x) + ", y:" + std::to_string(y) +
                ", z:" + std::to_string(z) + "}";
    }
};

cv::Mat operator*(const Rotation& lhs, const cv::Point3d& rhs)
{
    cv::Mat mat = cv::Mat(rhs).reshape(1);
    // Combined rotation matrix
    return lhs.toMat() * mat;
}
cv::Mat operator*(const RotationTranslation& lhs, const cv::Point3d& rhs)
{
    auto rotated = cv::Point3d(lhs.rot * rhs);
    rotated.x += lhs.x;
    rotated.y += lhs.y;
    rotated.z += lhs.z;
    return cv::Mat(rotated).reshape(1);
}

class AssessCalibration {

    public:
        AssessCalibration(): nh_("~") {
            
            nh_.getParam("visualise_pose_num", visualise_pose_num);
            nh_.getParam("visualise", visualise);
            nh_.getParam("csv", csv);

            const size_t last_slash_idx = csv.rfind('/');
            if (std::string::npos != last_slash_idx)
            {
                data_dir = csv.substr(0, last_slash_idx);
            }
            
            public_nh_.getParam("distortion_model", distortion_model);
            public_nh_.getParam("height", height);
            public_nh_.getParam("width", width);
            public_nh_.getParam("K", K);
            public_nh_.getParam("D", D);

            public_nh_.getParam("chessboard/board_dimension/width", board_dimensions.width);
            public_nh_.getParam("chessboard/board_dimension/height", board_dimensions.height);

            import_samples(data_dir + "/poses.csv");

            // Load in camera_info to cv::Mat
            cameramat = cv::Mat::zeros(3, 3, CV_64F);
            distcoeff = cv::Mat::eye(1, 4, CV_64F);
            cameramat.at<double>(0, 0) = K[0];
            cameramat.at<double>(0, 2) = K[2];
            cameramat.at<double>(1, 1) = K[4];
            cameramat.at<double>(1, 2) = K[5];
            cameramat.at<double>(2, 2) = 1;

            distcoeff.at<double>(0) = D[0];
            distcoeff.at<double>(1) = D[1];
            distcoeff.at<double>(2) = D[2];
            distcoeff.at<double>(3) = D[3];

            param_msg = ros::topic::waitForMessage<std_msgs::Float64MultiArray>("/extrinsic_calib_param");
            if(param_msg != NULL){
                param_msg_callback();
            }
        }

        // Get the mean and stdev from visualise_results.py
        void param_msg_callback() {

            // Need to inverse the transforms
            // In calibration, we figured out the transform to make camera into lidar frame (here we do opposite)
            // Here we apply transform to lidar i.e. tf(lidar) (parent) -----> camera (child)
            tf2::Transform transform;
            tf2::Quaternion quat;
            tf2::Vector3 trans;
            quat.setRPY(param_msg->data[0],param_msg->data[1],param_msg->data[2]);
            trans.setX(param_msg->data[3]);
            trans.setY(param_msg->data[4]);
            trans.setZ(param_msg->data[5]);
            transform.setRotation(quat);
            transform.setOrigin(trans);
            // ROS_INFO("Inverting rotation and translation for projecting LiDAR points into camera image");
            tf_msg.transform.rotation.w = transform.inverse().getRotation().w();
            tf_msg.transform.rotation.x = transform.inverse().getRotation().x();
            tf_msg.transform.rotation.y = transform.inverse().getRotation().y();
            tf_msg.transform.rotation.z = transform.inverse().getRotation().z();
            tf_msg.transform.translation.x = transform.inverse().getOrigin().x();
            tf_msg.transform.translation.y = transform.inverse().getOrigin().y();
            tf_msg.transform.translation.z = transform.inverse().getOrigin().z();
            
            double r_val,y_val,p_val;
            double d1,d2,d3;
            geometry_msgs::Quaternion q = tf_msg.transform.rotation;
            tf::Quaternion tfq;
            tf::quaternionMsgToTF(q, tfq);
            tf::Matrix3x3(tfq).getEulerYPR(y_val,p_val,r_val);
            rot_trans.x = tf_msg.transform.translation.x * 1000;
            rot_trans.y = tf_msg.transform.translation.y * 1000;
            rot_trans.z = tf_msg.transform.translation.z * 1000;
            rot_trans.rot.roll = r_val;
            rot_trans.rot.pitch = p_val;
            rot_trans.rot.yaw = y_val;

            results_and_visualise();
        }

        void results_and_visualise () 
        {    
            std::printf("\n---- Calculating average reprojection error on %d samples ---- \n", sample_list.size());
            // Calculate mean and stdev of pixel error across all test samples
            std::vector<float> pix_err, pix_errmm;
            for (int i = 0; i < sample_list.size(); i++)
            {
                std::vector<cv::Point2d> cam, lidar;
                float pe = compute_reprojection(sample_list[i], cam, lidar);
                pix_err.push_back(pe);
                pix_errmm.push_back(pe*sample_list[i].pixeltometre*1000);

                // Get board dimension error - hardcoded board dims for now
                double w0_diff = abs(sample_list[i].widths[0] - board_dimensions.width);
                double w1_diff = abs(sample_list[i].widths[1] - board_dimensions.width);
                double h0_diff = abs(sample_list[i].heights[0] - board_dimensions.height);
                double h1_diff = abs(sample_list[i].heights[1] - board_dimensions.height);
                double be_dim_err = w0_diff + w1_diff + h0_diff + h1_diff;
                std::printf(" %3d/%3d | dist=%6.3fm, dimerr=%8.3fmm | error: %7.3fpix  --> %7.3fmm\n", i+1, sample_list.size(), sample_list[i].distance_from_origin, be_dim_err, pe, pe*sample_list[i].pixeltometre*1000);
            }   
            float mean_pe, stdev_pe, mean_pemm, stdev_pemm;
            get_mean_stdev(pix_err, mean_pe, stdev_pe);
            get_mean_stdev(pix_errmm, mean_pemm, stdev_pemm);
            printf("\nCalibration params (roll,pitch,yaw,x,y,z): %6.4f,%6.4f,%6.4f,%6.4f,%6.4f,%6.4f\n", param_msg->data[0],param_msg->data[1],param_msg->data[2],param_msg->data[3],param_msg->data[4],param_msg->data[5]);
            printf("\nMean reprojection error across  %d samples\n", sample_list.size());
            std::printf("- Error (pix) = %6.3f pix, stdev = %6.3f\n", mean_pe, stdev_pe);
            std::printf("- Error (mm)  = %6.3f mm , stdev = %6.3f\n\n\n", mean_pemm, stdev_pemm);
            
            if (visualise)
            {
                std::string image_path = data_dir + "/images/pose" + std::to_string(visualise_pose_num) + ".png";
                std::string pcd_path = data_dir + "/pcd/pose" + std::to_string(visualise_pose_num) + "_full.pcd";
                
                //  Project the two centres onto an image
                std::vector<cv::Point2d> cam_project, lidar_project;
                cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
                 if (image.empty()) {
                    ROS_ERROR_STREAM("Could not read image file, check if image exists at: " << image_path);
                }

                pcl::PointCloud<pcl::PointXYZIR>::Ptr og_cloud(new pcl::PointCloud<pcl::PointXYZIR>);
                pcl::PointCloud<pcl::PointXYZIR>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZIR>);
                if(pcl::io::loadPCDFile<pcl::PointXYZIR> (pcd_path, *og_cloud) == -1)
                {
                    ROS_ERROR_STREAM("Could not read pcd file, check if pcd file exists at: " << pcd_path);
                } else {
                    sensor_msgs::PointCloud2 cloud_msg;
                    pcl::toROSMsg(*og_cloud, cloud_msg);

                    sensor_msgs::PointCloud2 cloud_tf;
                    tf2::doTransform(cloud_msg, cloud_tf, tf_msg);
                    pcl::fromROSMsg(cloud_tf, *cloud);

                    if ( cloud->points.size() ) {
                        
                        for (pcl::PointCloud<pcl::PointXYZIR>::const_iterator it = cloud->begin(); it != cloud->end(); it++) {
                            double tmpxC = it->x / it->z;
                            double tmpyC = it->y / it->z;
                            double tmpzC = it->z;
                            double dis = pow(it->x * it->x + it->y * it->y + it->z * it->z, 0.5);
                            cv::Point2d planepointsC;
                            int range = std::min(round((dis / 30.0) * 49), 49.0);

                            // Applying the distortion
                            double r2 = tmpxC * tmpxC + tmpyC * tmpyC;
                            double r1 = pow(r2, 0.5);
                            double a0 = std::atan(r1);
                            double a1;
                            a1 = a0 * (1 + distcoeff.at<double>(0) * pow(a0, 2) + distcoeff.at<double>(1) * pow(a0, 4) +
                                    distcoeff.at<double>(2) * pow(a0, 6) + distcoeff.at<double>(3) * pow(a0, 8));
                            planepointsC.x = (a1 / r1) * tmpxC;
                            planepointsC.y = (a1 / r1) * tmpyC;

                            planepointsC.x = cameramat.at<double>(0, 0) * planepointsC.x + cameramat.at<double>(0, 2);
                            planepointsC.y = cameramat.at<double>(1, 1) * planepointsC.y + cameramat.at<double>(1, 2);

                            if (planepointsC.y >= 0 and planepointsC.y < height and planepointsC.x >= 0 and planepointsC.x < width and
                                tmpzC >= 0 and std::abs(tmpxC) <= 1.35) {

                                int point_size = 2;
                                cv::circle(image,
                                    cv::Point(planepointsC.x, planepointsC.y), point_size,
                                    CV_RGB(255 * colmap[50-range][0], 255 * colmap[50-range][1], 255 * colmap[50-range][2]), -1);
                            }
                        }
                    }
                }
                ROS_INFO_STREAM("Projecting points onto image for pose #" << (visualise_pose_num));
                compute_reprojection(sample_list[visualise_pose_num-1], cam_project, lidar_project);
                for (auto& point : cam_project)
                {
                    cv::circle(image, point, 15, CV_RGB(0, 255, 0), 2);
                    cv::drawMarker(image, point, CV_RGB(0,255,0), cv::MARKER_CROSS, 25, 2, cv::LINE_8);
                }
                for (auto& point : lidar_project)
                {
                    cv::circle(image, point, 15, CV_RGB(255, 255, 0), 2);
                    cv::drawMarker(image, point, CV_RGB(255,255,0), cv::MARKER_TILTED_CROSS, 20, 2, cv::LINE_8);
                }
                cv::Mat resized_img;
                cv::resize(image, resized_img, cv::Size(), 0.75, 0.75);
                cv::imshow("Reprojection", resized_img);        
                cv::waitKey(0);
            }
        }

        void import_samples(std::string pose_path)
        {
            // Read row by row into a Point3d vector
            std::vector<cv::Point3d> row;
            std::string line, word;

            std::ifstream read_samples(pose_path);
            if (!read_samples.good()) 
            {
                ROS_ERROR_STREAM("REPROJECTION - No pose file found at " << pose_path);
            }
            ROS_INFO_STREAM("Importing samples from: " << pose_path);
            while (std::getline(read_samples, line, '\n')) {

                // used for breaking up words
                std::stringstream s(line);

                // read every column data of a row and store it in a string variable, 'word'
                std::vector<double> line_double;
                while (getline(s, word, ',')) {
                    line_double.push_back(atof(word.c_str()));
                }
                if (line_double.size() > 1){
                    row.push_back({line_double[0], line_double[1], line_double[2]});
                } else {
                    row.push_back({line_double[0], 0, 0});
                }
            }
            
            // Shove the double vector elements into the OptimiseSample struct
            int sample_numrows = 19; // non-zero indexed, but the i value is.
            for (int i = 0; i < row.size(); i+=sample_numrows) {
                OptimisationSample temp;
                temp.camera_centre = row[i];
                temp.camera_normal = row[i+1];
                for (int j = 0; j < 4; j++){
                    temp.camera_corners.push_back(row[i+2+j]);
                }
                temp.lidar_centre = row[i+6];
                temp.lidar_normal = row[i+7];
                for (int k = 0; k < 4; k++){
                    temp.lidar_corners.push_back(row[i+8+k]);
                }
                temp.angles_0.push_back(row[i+12].x);
                temp.angles_0.push_back(row[i+12].y);
                temp.angles_1.push_back(row[i+13].x);
                temp.angles_1.push_back(row[i+13].y);
                temp.widths.push_back(row[i+14].x);
                temp.widths.push_back(row[i+14].y);
                temp.heights.push_back(row[i+15].x);
                temp.heights.push_back(row[i+15].y);
                temp.distance_from_origin = row[i+16].x;
                temp.pixeltometre = row[i+17].x;
                temp.sample_num = row[i+sample_numrows-1].x;
                sample_list.push_back(temp);
            }

            read_samples.close();
            ROS_INFO_STREAM(sample_list.size() << " samples imported");
        }

        float compute_reprojection(OptimisationSample sample, std::vector<cv::Point2d> &cam, std::vector<cv::Point2d> &lidar)
        {
            cv::Mat rvec = cv::Mat_<double>::zeros(3, 1);
            cv::Mat tvec = cv::Mat_<double>::zeros(3, 1);

            std::vector<cv::Point3d> cam_centre_3d;
            std::vector<cv::Point3d> lidar_centre_3d;

            // Need to rotate the lidar points to the camera frame if we want to project it into the image 
            // Cause otherwise, projectPoints function doesn't know how to project points that are in the lidar frame (duh!)
            cv::Point3d lidar_centre_camera_frame = cv::Point3d(rot_trans * sample.lidar_centre);
            cam_centre_3d.push_back(sample.camera_centre);
            lidar_centre_3d.push_back(lidar_centre_camera_frame);

            // ROS_INFO_STREAM("Camera distortion model = " << distortion_model);
            std::vector<cv::Point2d> cam_dist, lidar_dist;

            if (distortion_model == "fisheye")
            {
                cv::fisheye::projectPoints(cam_centre_3d, cam, rvec, tvec, cameramat, distcoeff);
                cv::fisheye::projectPoints(lidar_centre_3d, lidar, rvec, tvec, cameramat, distcoeff);
            }
            else
            {
                cv::projectPoints(cam_centre_3d, rvec, tvec, cameramat, distcoeff, cam);
                cv::projectPoints(lidar_centre_3d, rvec, tvec, cameramat, distcoeff, lidar);
            }

            float pixel_error = cv::norm(cam[0] - lidar[0]);
            return pixel_error;
        }

        void get_mean_stdev(std::vector<float>& input_vec, float& mean, float& stdev)
        {
            float sum = std::accumulate(std::begin(input_vec), std::end(input_vec), 0.0);
            mean =  sum / input_vec.size();

            float accum = 0.0;
            std::for_each (std::begin(input_vec), std::end(input_vec), [&](const float d) {
                accum += (d - mean) * (d - mean);
            });

            stdev = sqrt(accum / (input_vec.size()-1));
        }

        

    private:
        ros::NodeHandle public_nh_;
        ros::NodeHandle nh_;

        std::string distortion_model;
        std::vector<double> K, D;
        cv::Mat cameramat, distcoeff;
        int height, width;
        cv::Size board_dimensions;

        std::string csv, data_dir;;
        int visualise_pose_num;
        bool visualise;

        ros::Subscriber extrinsic_calib_param_sub_;   
        std_msgs::Float64MultiArray::ConstPtr param_msg;
        std::vector<OptimisationSample> sample_list;
        geometry_msgs::TransformStamped tf_msg;
        RotationTranslation rot_trans;  
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "assess_calibration");
    AssessCalibration ac;
    ros::spin();
}

    