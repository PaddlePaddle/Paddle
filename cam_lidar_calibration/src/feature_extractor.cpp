#include "cam_lidar_calibration/feature_extractor.h"

#include <list>

#include <ros/ros.h>

#include "cam_lidar_calibration/point_xyzir.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/intersections.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/extract_indices.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/impl/project_inliers.hpp>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/impl/sac_segmentation.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Geometry>
#include <cmath>

#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/MarkerArray.h>

#include <ros/package.h>

// For shuffling of generated sets
#include <algorithm>

using cv::findChessboardCorners;
using cv::Mat_;
using cv::Size;
using cv::TermCriteria;
using PointCloud = pcl::PointCloud<pcl::PointXYZIR>;

namespace cam_lidar_calibration
{
    FeatureExtractor::FeatureExtractor()
    {
        // Creating ROS nodehandle
        private_nh = ros::NodeHandle("~");
        public_nh = ros::NodeHandle();
        ros::NodeHandle pnh = ros::NodeHandle("~");  // getMTPrivateNodeHandle();
        private_nh.getParam("import_path", import_path);
        private_nh.getParam("import_samples", import_samples);
        private_nh.getParam("num_lowestvoq", num_lowestvoq);
        private_nh.getParam("distance_offset_mm", distance_offset);
        loadParams(public_nh, i_params);
        optimiser_ = std::make_shared<Optimiser>(i_params);
        ROS_INFO("Input parameters loaded");

        it_.reset(new image_transport::ImageTransport(public_nh));
        it_p_.reset(new image_transport::ImageTransport(private_nh));

        // Dynamic reconfigure gui to set the experimental region bounds
        server = boost::make_shared<dynamic_reconfigure::Server<cam_lidar_calibration::boundsConfig>>(pnh);
        dynamic_reconfigure::Server<cam_lidar_calibration::boundsConfig>::CallbackType f;
        f = boost::bind(&FeatureExtractor::boundsCB, this, _1, _2);
        server->setCallback(f);

        // Synchronizer to get synchronized camera-lidar scan pairs
        image_sub_ = std::make_shared<image_sub_type>(private_nh, i_params.camera_topic, queue_rate_);
        pc_sub_ = std::make_shared<pc_sub_type>(private_nh, i_params.lidar_topic, queue_rate_);

        image_pc_sync_ = std::make_shared<message_filters::Synchronizer<ImageLidarSyncPolicy>>(
                ImageLidarSyncPolicy(queue_rate_), *image_sub_, *pc_sub_);
        image_pc_sync_->registerCallback(boost::bind(&FeatureExtractor::extractRegionOfInterest, this, _1, _2));

        board_cloud_pub_ = private_nh.advertise<PointCloud>("chessboard", 1);
        bounded_cloud_pub_ = private_nh.advertise<PointCloud>("experimental_region", 10);
        optimise_service_ = public_nh.advertiseService("optimiser", &FeatureExtractor::serviceCB, this);
        samples_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("collected_samples", 0);
        image_publisher = it_->advertise("camera_features", 1);

        valid_camera_info = false;
        i_params.cameramat = cv::Mat::zeros(3, 3, CV_64F);
        i_params.distcoeff = cv::Mat::eye(1, 4, CV_64F);
        camera_info_sub_ = public_nh.subscribe(i_params.camera_info, 20, &FeatureExtractor::callback_camerainfo, this);

        // Create folder for output if it does not exist
        curdatetime = getDateTime();

        // Specify folder for saving samples
        if (import_samples){
            std::string import_dir;
            const size_t last_slash_idx = import_path.rfind('/');
            if (std::string::npos != last_slash_idx)
            {
                newdatafolder = import_path.substr(0, last_slash_idx);
            }
        } else{
            std::string data_dir = ros::package::getPath("cam_lidar_calibration") + "/data";
            // Successful capture, store jpeg and pcd file
            if(boost::filesystem::create_directory(data_dir))
            {
                ROS_INFO_STREAM("Data save folder created at " << data_dir);
            }
            newdatafolder = data_dir + "/" + curdatetime;
        }
        
        ROS_INFO("Finished init cam_lidar_calibration");
    }

    void FeatureExtractor::callback_camerainfo(const sensor_msgs::CameraInfo::ConstPtr &msg) {

        i_params.cameramat.at<double>(0, 0) = msg->K[0];
        i_params.cameramat.at<double>(0, 2) = msg->K[2];
        i_params.cameramat.at<double>(1, 1) = msg->K[4];
        i_params.cameramat.at<double>(1, 2) = msg->K[5];
        i_params.cameramat.at<double>(2, 2) = 1;

        i_params.distcoeff.at<double>(0) = msg->D[0];
        i_params.distcoeff.at<double>(1) = msg->D[1];
        i_params.distcoeff.at<double>(2) = msg->D[2];
        i_params.distcoeff.at<double>(3) = msg->D[3];

        i_params.image_size = std::make_pair(msg->width, msg->height);

        // Fisheye/equidistant
        if (msg->distortion_model == "equidistant") {
            i_params.fisheye_model = true;
        // Pinhole
        } else if (msg->distortion_model == "rational_polynomial" or msg->distortion_model == "plumb_bob") {
            i_params.fisheye_model = false;
        } else {
            ROS_FATAL_STREAM("Camera model " << msg->distortion_model << " not supported");
        }
        valid_camera_info = true;
    }

    bool FeatureExtractor::serviceCB(Optimise::Request& req, Optimise::Response& res)
    {
        switch (req.operation)
        {
            case Optimise::Request::CAPTURE:
                ROS_INFO("Capturing sample");
                break;
            case Optimise::Request::DISCARD:
                ROS_INFO("Discarding last sample");
                if (!optimiser_->samples.empty())
                {
                    num_samples--;
                    optimiser_->samples.pop_back();
                    pc_samples_.pop_back();
                }
                break;
        }
        publishBoardPointCloud();
        flag = req.operation;  // read flag published by rviz calibration panel
        // Wait for operation to complete
        while (flag == Optimise::Request::CAPTURE)
        {
        }
        res.samples = optimiser_->samples.size();
        return true;
    }

    bool compare_voq(const SetAssess &a, const SetAssess &b)
    {
        return a.voq < b.voq;
    }

    void FeatureExtractor::optimise(const RunOptimiseGoalConstPtr& goal,
                                    actionlib::SimpleActionServer<RunOptimiseAction>* as)
    {
        ROS_INFO("Starting FeatureExtractor::optimise");

        std::string curdatetime = getDateTime();

        if (import_samples) {
            ROS_INFO_STREAM("Reading file: " << import_path);
            std::ifstream read_samples(import_path);
            

            optimiser_->samples.resize(0);

            // Read row by row into a Point3d vector
            std::vector<cv::Point3d> row;
            std::string line, word;

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
                optimiser_->samples.push_back(temp);
            }

            read_samples.close();
            ROS_INFO_STREAM(optimiser_->samples.size() << " samples imported");
        } else {

            std::string savesamplespath = newdatafolder + "/poses.csv";
            std::ofstream save_samples;
            save_samples.open(savesamplespath, std::ios_base::out | std::ios_base::trunc) ;

            for (OptimisationSample s : optimiser_->samples){
                save_samples << s.camera_centre.x << "," << s.camera_centre.y << "," << s.camera_centre.z << "\n";
                save_samples << s.camera_normal.x << "," << s.camera_normal.y << "," << s.camera_normal.z << "\n";
                for (auto cc : s.camera_corners) {
                    save_samples << cc.x << "," << cc.y << "," << cc.z << "\n";
                }
                save_samples << s.lidar_centre.x << "," << s.lidar_centre.y << "," << s.lidar_centre.z << "\n";
                save_samples << s.lidar_normal.x << "," << s.lidar_normal.y << "," << s.lidar_normal.z << "\n";
                for (auto lc : s.lidar_corners) {
                    save_samples << lc.x << "," << lc.y << "," << lc.z << "\n";
                }
                save_samples << s.angles_0[0] << "," << s.angles_0[1] << ",0\n";
                save_samples << s.angles_1[0] << "," << s.angles_1[1] << ",0\n";
                save_samples << s.widths[0] << "," << s.widths[1] << ",0\n";
                save_samples << s.heights[0] << "," << s.heights[1] << ",0\n";
                save_samples << s.distance_from_origin << ",0,0\n";
                save_samples << s.pixeltometre << ",0,0\n";
                save_samples << s.sample_num << ",0,0\n";
            }
            save_samples.close();
            ROS_INFO_STREAM("Samples written to file: " << savesamplespath);
            ROS_INFO_STREAM("All " << optimiser_->samples.size() << " samples saved");
        }
        if (optimiser_->samples.size() < 3){
            ROS_ERROR("Less than 3 samples captured or imported.");
            return;
        }

        RotationTranslation opt_result;
        bool success;
        EA::Chronometer timer_all, timer_set, timer_assess;
        timer_all.tic();
        timer_assess.tic();

        // Generate all N choose 3 combinations 
        std::vector<OptimisationSample> set;

        // If less than 100 samples, we can get all 100C3, any more samples and NC3 grows too large so we just randomly sample for speed
        if (optimiser_->samples.size() < 100) {
            optimiser_->generate_sets(0, 3, set, optimiser_->samples);
        } else {
            for (int j = 0; j < 19600; j++) {
                for (int i = 0; i < 3; i++) {

                    // Check if sample already exists in set
                    OptimisationSample new_sample;

                    int rnd_snum = rand() % optimiser_->samples.size();
                    new_sample = optimiser_->samples[rnd_snum];
                    int s_num = new_sample.sample_num;
                    auto it = std::find_if(set.begin(), set.end(), [&s_num](const OptimisationSample& obj) {return obj.sample_num == s_num;});
                    
                    while(it != set.end())
                    {                                
                        int rnd_snum2 = rand() % optimiser_->samples.size();
                        new_sample = optimiser_->samples[rnd_snum2];
                        it = std::find_if(set.begin(), set.end(), [&new_sample](const OptimisationSample& obj) {return obj.sample_num == new_sample.sample_num;});
                    }
                    set.push_back(new_sample);
                }
                optimiser_->sets.push_back(set);
                set.resize(0);
            }
        }

        std::srand(std::time(0));
        std::random_shuffle(optimiser_->sets.begin(), optimiser_->sets.end());

        // Generate the top num_lowestvoq sets of lowest VOQ scores
        int num_assessed = 0;
        std::vector<SetAssess> calib_list;

        for (int i = 0; i < optimiser_->sets.size(); i++)
        {
            // Insert vector elements into matrix to compute analytical euler angles by matrix operations
            int row = 0;
            auto camera_centres_ = cv::Mat(optimiser_->sets[i].size(), 3, CV_64F);
            auto camera_normals_ = cv::Mat(optimiser_->sets[i].size(), 3, CV_64F);
            auto lidar_centres_ = cv::Mat(optimiser_->sets[i].size(), 3, CV_64F);
            auto lidar_normals_ = cv::Mat(optimiser_->sets[i].size(), 3, CV_64F);
            std::vector<float> be;

            for (auto& sample : optimiser_->sets[i])
            {
                float err_dim = abs(sample.widths[0] - i_params.board_dimensions.width)+abs(sample.widths[1] - i_params.board_dimensions.width)+abs(sample.heights[0] - i_params.board_dimensions.height)+abs(sample.heights[1] - i_params.board_dimensions.height);
                be.push_back(err_dim);

                cv::Mat cn = cv::Mat(sample.camera_normal).reshape(1).t();
                cn.copyTo(camera_normals_.row(row));
                cv::Mat cc = cv::Mat(sample.camera_centre).reshape(1).t();
                cc.copyTo(camera_centres_.row(row));
                cv::Mat ln = cv::Mat(sample.lidar_normal).reshape(1).t();
                ln.copyTo(lidar_normals_.row(row));
                cv::Mat lc = cv::Mat(sample.lidar_centre).reshape(1).t();
                lc.copyTo(lidar_centres_.row(row));
                row++;
            }

            float b_avg = std::accumulate(std::begin(be), std::end(be), 0.0)/be.size();

            // Commutative property holds for AA^{-1} = A^{-1}A = I (in the case of a well conditioned matrix)
            float cn_cond_fro = cv::norm(camera_normals_, cv::NORM_L2) * cv::norm(camera_normals_.inv(), cv::NORM_L2);
            float ln_cond_fro = cv::norm(lidar_normals_, cv::NORM_L2) * cv::norm(lidar_normals_.inv(), cv::NORM_L2);
            float cond_max = (cn_cond_fro > ln_cond_fro) ? cn_cond_fro : ln_cond_fro;
            float voq = cond_max + b_avg;

            SetAssess new_set;
            new_set.voq = voq;
            new_set.set = optimiser_->sets[i];

            // calib_list is a list of size 50 that maintains the lowest VOQ values 
            // by keeping track of its max element, and replacing that with the next lowest VOQ.
            if (calib_list.size() < num_lowestvoq)
            {
                calib_list.push_back(new_set);
                if (calib_list.size() == num_lowestvoq)
                {
                    // sort such that the last element is the max
                    std::sort(calib_list.begin(), calib_list.end(), compare_voq);
                }
            } else {

                // Compare new element with max element (which is the last element)
                if (new_set.voq < calib_list.back().voq) {
                    calib_list.pop_back();
                    calib_list.push_back(new_set);
                    std::sort(calib_list.begin(), calib_list.end(), compare_voq);
                }
            }
            num_assessed++;
        }
        // Populate the optimiser sets with the top sets
        for (SetAssess sa : calib_list)
        {
            optimiser_->top_sets.push_back(sa.set);
        }
        ROS_INFO_STREAM("voq range: " << calib_list.front().voq << "-" << calib_list.back().voq);
        ROS_INFO_STREAM("Number of assessed sets: " << num_assessed);
        ROS_INFO_STREAM(optimiser_->top_sets.size() << " selected sets for optimisation");
        ROS_INFO_STREAM("Time taken: " << timer_assess.toc() << "s ");

        std::ofstream output_csv;
        std::string outpath = newdatafolder + "/calibration_" + curdatetime + ".csv";
        ROS_INFO_STREAM("Calibration results will be saved at: " << outpath);
        output_csv.open(outpath, std::ios_base::out | std::ios_base::trunc);
        output_csv << "roll,pitch,yaw,x,y,z\n";
        output_csv.close();

        ROS_INFO("====== START CALIBRATION ======\n");

        printf(" Computing calibration results (roll,pitch,yaw,x,y,z) for each of the %d lowest voq sets\n", optimiser_->top_sets.size());
        for (int i = 0; i < optimiser_->top_sets.size(); i++)
        {
            output_csv.open(outpath, std::ios_base::ate | std::ios_base::app);

            timer_set.tic();
            timer_set.tic();
            printf(" %2d/%2d ", i+1, optimiser_->top_sets.size());
            success = optimiser_->optimise(opt_result, optimiser_->top_sets[i], i_params.cameramat, i_params.distcoeff);

            // Save extrinsic params to csv for post processing
            if (success) {

                output_csv << opt_result.rot.roll << "," << opt_result.rot.pitch << "," << opt_result.rot.yaw << ","
                        << opt_result.x / 1000.0 << "," << opt_result.y / 1000.0 << "," << opt_result.z / 1000.0 << "\n";
            }
            printf("| t: %.3fs\n", timer_set.toc());
            output_csv.close();
        }
        std::cout << "Optimisation Completed in " << timer_all.toc() << "s\n" << std::endl;
        ROS_INFO("====== END ======");

        ros::shutdown();
        return;

        // // Not in use
        // RunOptimiseResult res;
        // res.transform.translation.x = opt_result.x / 1000.;
        // res.transform.translation.y = opt_result.y / 1000.;
        // res.transform.translation.z = opt_result.z / 1000.;
        // Eigen::Matrix3d mat;
        // cv::cv2eigen(opt_result.rot.toMat(), mat);
        // Eigen::Quaterniond quat(mat);
        // tf::quaternionEigenToMsg(quat, res.transform.rotation);
        // as->setSucceeded(res);
    }

    void FeatureExtractor::publishBoardPointCloud()
    {
        // Publish collected board clouds
        PointCloud pc;
        pc.header.frame_id = lidar_frame_;
        for (auto board : pc_samples_)
        {
            pc += *board;
        }
        board_cloud_pub_.publish(pc);
    }

    void FeatureExtractor::boundsCB(cam_lidar_calibration::boundsConfig& config, uint32_t level)
    {
        // Read the values corresponding to the motion of slider bars in reconfigure gui
        bounds_ = config;
        ROS_INFO("Reconfigure Request: %lf %lf %lf %lf %lf %lf", config.x_min, config.x_max, config.y_min, config.y_max,
                 config.z_min, config.z_max);
    }

    geometry_msgs::Quaternion normalToQuaternion(const cv::Point3d& normal)
    {
        // Convert to Eigen vector
        Eigen::Vector3d eigen_normal(normal.x, normal.y, normal.z);
        Eigen::Vector3d axis(1, 0, 0);
        auto eigen_quat = Eigen::Quaterniond::FromTwoVectors(axis, eigen_normal);
        geometry_msgs::Quaternion quat;
        quat.w = eigen_quat.w();
        quat.x = eigen_quat.x();
        quat.y = eigen_quat.y();
        quat.z = eigen_quat.z();
//        std::cout << "calculating rotation between v0=[" << axis.x() << "," << axis.y() << "," << axis.z() << "], and v1=[";
//        std::cout << normal.x << "," << normal.y << "," << normal.z << "]" << std::endl;
//        std::cout << "quaternion(w,x,y,z): " << quat.w << "," << quat.x << "," << quat.y << "," << quat.z << std::endl;
        return quat;
    }

    void FeatureExtractor::visualiseSamples()
    {
        visualization_msgs::MarkerArray vis_array;

        int id = 1;
        visualization_msgs::Marker clear;
        clear.action = visualization_msgs::Marker::DELETEALL;
        vis_array.markers.push_back(clear);
        for (auto& sample : optimiser_->samples)
        {
            visualization_msgs::Marker lidar_board, lidar_normal;

            lidar_board.header.frame_id = lidar_normal.header.frame_id = lidar_frame_;
            lidar_board.action = lidar_normal.action = visualization_msgs::Marker::ADD;
            lidar_board.type = visualization_msgs::Marker::LINE_STRIP;
            lidar_normal.type = visualization_msgs::Marker::ARROW;

            lidar_normal.scale.x = 0.5;
            lidar_normal.scale.y = 0.04;
            lidar_normal.scale.z = 0.04;
            lidar_normal.color.a = 1.0;
            lidar_normal.color.b = 1.0;
            lidar_normal.color.g = 0.0;
            lidar_normal.color.r = 0.0;
            lidar_normal.pose.position.x = sample.lidar_centre.x / 1000;
            lidar_normal.pose.position.y = sample.lidar_centre.y / 1000;
            lidar_normal.pose.position.z = sample.lidar_centre.z / 1000;
            lidar_normal.pose.orientation = normalToQuaternion(sample.lidar_normal);
            lidar_normal.id = id++;

            vis_array.markers.push_back(lidar_normal);

            lidar_board.scale.x = 0.01;
            lidar_board.scale.y = 0.01;
            lidar_board.scale.z = 0.01;
            lidar_board.color.a = 1.0;
            lidar_board.color.b = 0.0;
            lidar_board.color.g = 1.0;
            lidar_board.color.r = 0.0;
            lidar_board.pose.orientation.w = 1.0;
            for (auto& c : sample.lidar_corners)
            {
                geometry_msgs::Point p;
                p.x = c.x / 1000;
                p.y = c.y / 1000;
                p.z = c.z / 1000;
                lidar_board.points.push_back(p);
            }
            lidar_board.points.push_back(lidar_board.points[0]);
            lidar_board.id = id++;
            vis_array.markers.push_back(lidar_board);
        }
        samples_pub_.publish(vis_array);
    }

    void FeatureExtractor::passthrough(const PointCloud::ConstPtr& input_pc, PointCloud::Ptr& output_pc)
    {
        PointCloud::Ptr x(new PointCloud);
        PointCloud::Ptr z(new PointCloud);
        // Filter out the experimental region
        pcl::PassThrough<pcl::PointXYZIR> pass;
        pass.setInputCloud(input_pc);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(bounds_.x_min, bounds_.x_max);
        pass.filter(*x);
        pass.setInputCloud(x);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(bounds_.z_min, bounds_.z_max);
        pass.filter(*z);
        pass.setInputCloud(z);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(bounds_.y_min, bounds_.y_max);
        pass.filter(*output_pc);
    }

    auto FeatureExtractor::chessboardProjection(const std::vector<cv::Point2d>& corners,
                                                const cv_bridge::CvImagePtr& cv_ptr)
    {
        // Find the chessboard in 3D space - in it's own object frame (position is arbitrary, so we place it flat)

        // Location of board frame origin from the bottom left inner corner of the chessboard
        cv::Point3d chessboard_bleft_corner((i_params.chessboard_pattern_size.width - 1) * i_params.square_length / 2,
                                      (i_params.chessboard_pattern_size.height - 1)*i_params.square_length/2, 0);

        std::vector<cv::Point3d> corners_3d;
        for (int y = 0; y < i_params.chessboard_pattern_size.height; y++)
        {
            for (int x = 0; x < i_params.chessboard_pattern_size.width; x++)
            {
                corners_3d.push_back(cv::Point3d(x, y, 0) * i_params.square_length - chessboard_bleft_corner);
            }
        }

        // chessboard corners, middle square corners, board corners and centre
        std::vector<cv::Point3d> board_corners_3d;
        // Board corner coordinates from the centre of the chessboard
        board_corners_3d.push_back(cv::Point3d((i_params.board_dimensions.width - i_params.cb_translation_error.x)/2.0,
                                                (i_params.board_dimensions.height - i_params.cb_translation_error.y)/2.0,0.0));

        board_corners_3d.push_back(cv::Point3d(-(i_params.board_dimensions.width + i_params.cb_translation_error.x)/2.0,
                                               (i_params.board_dimensions.height - i_params.cb_translation_error.y)/2.0,0.0));

        board_corners_3d.push_back(cv::Point3d(-(i_params.board_dimensions.width + i_params.cb_translation_error.x)/2.0,
                                               -(i_params.board_dimensions.height + i_params.cb_translation_error.y)/2.0,0.0));

        board_corners_3d.push_back(cv::Point3d((i_params.board_dimensions.width - i_params.cb_translation_error.x)/2.0,
                                               -(i_params.board_dimensions.height + i_params.cb_translation_error.y)/2.0,0.0));
        // Board centre coordinates from the centre of the chessboard (due to incorrect placement of chessboard on board)
        board_corners_3d.push_back(cv::Point3d(-i_params.cb_translation_error.x/2.0, -i_params.cb_translation_error.y/2.0, 0.0));

        std::vector<cv::Point2d> inner_cbcorner_pixels, board_image_pixels;
        cv::Mat rvec(3, 3, cv::DataType<double>::type);  // Initialization for pinhole and fisheye cameras
        cv::Mat tvec(3, 1, cv::DataType<double>::type);

        if (valid_camera_info) {
            if (i_params.fisheye_model)
            {
                // Undistort the image by applying the fisheye intrinsic parameters
                // the final input param is the camera matrix in the new or rectified coordinate frame.
                // We put this to be the same as i_params_.cameramat or else it will be set to empty matrix by default.
                std::vector<cv::Point2d> corners_undistorted;
                cv::fisheye::undistortPoints(corners, corners_undistorted, i_params.cameramat, i_params.distcoeff,
                                             i_params.cameramat);
                cv::solvePnP(corners_3d, corners_undistorted, i_params.cameramat, cv::noArray(), rvec, tvec);
                cv::fisheye::projectPoints(corners_3d, inner_cbcorner_pixels, rvec, tvec, i_params.cameramat, i_params.distcoeff);
                cv::fisheye::projectPoints(board_corners_3d, board_image_pixels, rvec, tvec, i_params.cameramat,
                                           i_params.distcoeff);
            } else {
                // Pinhole model
                cv::solvePnP(corners_3d, corners, i_params.cameramat, i_params.distcoeff, rvec, tvec);
                cv::projectPoints(corners_3d, rvec, tvec, i_params.cameramat, i_params.distcoeff, inner_cbcorner_pixels);
                cv::projectPoints(board_corners_3d, rvec, tvec, i_params.cameramat, i_params.distcoeff, board_image_pixels);
            }
        } else {
            ROS_FATAL("No msgs from /camera_info - check camera_info topic in cfg/params.yaml is correct and is being published");
        }

        for (int i = 0; i < board_image_pixels.size(); i++){
            if (i == 0){
                cv::circle(cv_ptr->image, board_image_pixels[i], 4, CV_RGB(255, 0, 0), -1);
            } else if (i == 1) {
                cv::circle(cv_ptr->image, board_image_pixels[i], 4, CV_RGB(0, 255, 0), -1);
            } else if (i == 2) {
                cv::circle(cv_ptr->image, board_image_pixels[i], 4, CV_RGB(0, 0, 255), -1);
            } else if (i == 3) {
                cv::circle(cv_ptr->image, board_image_pixels[i], 4, CV_RGB(255, 255, 0), -1);
            } else if (i == 4) {
                cv::circle(cv_ptr->image, board_image_pixels[i], 4, CV_RGB(0, 255, 255), -1);
            }
        }

        for (auto& point : inner_cbcorner_pixels)
        {
            cv::circle(cv_ptr->image, point, 3, CV_RGB(255, 0, 0), -1);
        }

        double pixdiagonal = sqrt(pow(inner_cbcorner_pixels.front().x-inner_cbcorner_pixels.back().x,2)+(pow(inner_cbcorner_pixels.front().y-inner_cbcorner_pixels.back().y,2)));
        double len_diagonal = sqrt(pow(corners_3d.front().x-corners_3d.back().x,2)+(pow(corners_3d.front().y-corners_3d.back().y,2)));
        metreperpixel_cbdiag = len_diagonal /(1000*pixdiagonal);

        // Return all the necessary coefficients
        return std::make_tuple(rvec, tvec, board_corners_3d);
    }

    std::tuple<std::vector<cv::Point3d>, cv::Mat>
    FeatureExtractor::locateChessboard(const sensor_msgs::Image::ConstPtr& image)
    {
        // Convert to OpenCV image object
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);
        std::vector<cv::Point2f> cornersf;
        std::vector<cv::Point2d> corners;
        // Find chessboard pattern in the image
        bool pattern_found = findChessboardCorners(gray, i_params.chessboard_pattern_size, cornersf,
                                                   cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        if (!pattern_found)
        {
            ROS_WARN("No chessboard found");
            std::vector<cv::Point3d> empty_corners;
            cv::Mat empty_normal;
            return std::make_tuple(empty_corners, empty_normal);
        }
        ROS_INFO("Chessboard found");
        // Find corner points with sub-pixel accuracy
        // This throws an exception if the corner points are doubles and not floats!?!
        cornerSubPix(gray, cornersf, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

        for (auto& corner : cornersf)
        {
            corners.push_back(cv::Point2d(corner));
        }

        auto [rvec, tvec, board_corners_3d] = chessboardProjection(corners, cv_ptr);
        
        cv::Mat rmat;
        cv::Rodrigues(rvec, rmat);
        cv::Mat z = cv::Mat(cv::Point3d(0., 0., -1.)); // TODO: why is this normal -1 in z? Surabhi's is just 1
        auto chessboard_normal = rmat * z;

        std::vector<cv::Point3d> corner_vectors;
        for (auto& corner : board_corners_3d)
        {   
            cv::Mat m(rmat * cv::Mat(corner).reshape(1) + tvec);
            corner_vectors.push_back(cv::Point3d(m));
        }

        // Publish the image with all the features marked in it
        ROS_INFO("Publishing chessboard image");
        image_publisher.publish(cv_ptr->toImageMsg());
        return std::make_tuple(corner_vectors, chessboard_normal);
    }

    std::tuple<pcl::PointCloud<pcl::PointXYZIR>::Ptr, cv::Point3d>
    FeatureExtractor::extractBoard(const PointCloud::Ptr& cloud, OptimisationSample &sample)
    {
        PointCloud::Ptr cloud_filtered(new PointCloud);
        // Filter out the board point cloud
        // find the point with max height(z val) in cloud_passthrough
        pcl::PointXYZIR cloud_min, cloud_max;
        pcl::getMinMax3D(*cloud, cloud_min, cloud_max);
        double z_max = cloud_max.z;
        // subtract by approximate diagonal length (in metres)
        double diag = std::hypot(i_params.board_dimensions.height, i_params.board_dimensions.width) /
                      1000.0;  // board dimensions are in mm
        double z_min = z_max - diag;
        pcl::PassThrough<pcl::PointXYZIR> pass_z;
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min, z_max);
        pass_z.setInputCloud(cloud);
        pass_z.filter(*cloud_filtered);  // board point cloud

        // Fit a plane through the board point cloud
        // Inliers give the indices of the points that are within the RANSAC threshold
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::SACSegmentation<pcl::PointXYZIR> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.004);
        pcl::ExtractIndices<pcl::PointXYZIR> extract;
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        // Check that segmentation succeeded
        PointCloud::Ptr cloud_projected(new PointCloud);
        if (coefficients->values.size() < 3)
        {
            ROS_WARN("Chessboard plane segmentation failed");
            cv::Point3d null_normal;
            return std::make_tuple(cloud_projected, null_normal);
        }

        // Plane normal vector magnitude
        cv::Point3d lidar_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        lidar_normal /= -cv::norm(lidar_normal);  // Normalise and flip the direction

        // Project the inliers on the fitted plane
        // When it freezes the chessboard after capture, what you see are the inlier points (filtered from the original)
        pcl::ProjectInliers<pcl::PointXYZIR> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(cloud_filtered);
        proj.setModelCoefficients(coefficients);
        proj.filter(*cloud_projected);


        // Publish the projected inliers
        pc_samples_.push_back(cloud_projected);
        // publishBoardPointCloud();
        return std::make_tuple(cloud_projected, lidar_normal);
    }

    std::pair<pcl::ModelCoefficients, pcl::ModelCoefficients>
    FeatureExtractor::findEdges(const PointCloud::Ptr& edge_pair_cloud)
    {
        pcl::ModelCoefficients full_coeff, half_coeff;
        pcl::PointIndices::Ptr full_inliers(new pcl::PointIndices), half_inliers(new pcl::PointIndices);
        PointCloud::Ptr half_cloud(new PointCloud);

        pcl::SACSegmentation<pcl::PointXYZIR> seg;
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02);
        seg.setInputCloud(edge_pair_cloud);
        seg.segment(*full_inliers, full_coeff);  // Fitting line1 through all points

         // Failed RANSAC returns empty coeffs
        if (full_coeff.values.empty()) {
            return std::make_pair(full_coeff, full_coeff);
        }

        pcl::ExtractIndices<pcl::PointXYZIR> extract;
        extract.setInputCloud(edge_pair_cloud);
        extract.setIndices(full_inliers);
        extract.setNegative(true);
        extract.filter(*half_cloud);
        seg.setInputCloud(half_cloud);
        seg.segment(*half_inliers, half_coeff);
        
         // Failed RANSAC returns empty coeffs        
        if (half_coeff.values.empty()) {
            return std::make_pair(half_coeff, half_coeff);
        }

        // Fitting line2 through outlier points
        // Determine which is above the other
        pcl::PointXYZIR full_min, full_max, half_min, half_max;
        pcl::getMinMax3D(*edge_pair_cloud, full_min, full_max);
        pcl::getMinMax3D(*half_cloud, half_min, half_max);
        if (full_max.z > half_max.z)
        {
            return std::make_pair(full_coeff, half_coeff);
        }
        else
        {
            return std::make_pair(half_coeff, full_coeff);
        }
    }

    void FeatureExtractor::distoffset_passthrough(const PointCloud::ConstPtr& input_pc, PointCloud::Ptr& output_pc)
    {
        if (distance_offset != 0) {
            pcl::PointCloud<pcl::PointXYZIR>::Ptr distoffset_pcl(new pcl::PointCloud<pcl::PointXYZIR>);
            distoffset_pcl->header = input_pc->header;
            for (auto p : input_pc->points)
            {
                // points are in metres (though you might see quite large nums that seem like cm)
                // convert cartesian (x,y,z) to polar (using spherical system)
                float x = p.x;
                float y = p.y;
                float z = p.z;

                if (x == 0 && y == 0 && z == 0)
                {
                    continue;
                }
                float r = sqrt(x * x + y * y + z * z);
                float theta = acosf(z / r); // inclination (vertical) - angle between positive z-axis and line from origin to point
                float phi = atanf(y / x);	// azimuth (horizontal) - angle between positive x-axis and line from origin to point on xy-plane

                // atan has the range [-90,90], so if you think of 4 quadrants, tan in quadrant 2 = quadrant 4, quadrant 1 = quadrant 3
                // Hence why the conversion back to cartesian just assume quad 4 is quad 2, and quad 3 is in quad 1
                int octant_before = find_octant(x, y, z);
                if (octant_before == 2 || octant_before == 6 || octant_before == 3 || octant_before == 7)
                {
                    phi = M_PI + phi;
                }

                // add distance offset (convert mm to metres)
                float r_do = r + distance_offset / 1000;

                // convert back to cartesian
                pcl::PointXYZIR point;
                point.x = r_do * sinf(theta) * cosf(phi);
                point.y = r_do * sinf(theta) * sinf(phi);
                point.z = r_do * cosf(theta);
                point.ring = p.ring;
                point.intensity = p.intensity;
                distoffset_pcl->push_back(point);
            }
            passthrough(distoffset_pcl, output_pc);
        } else {
            passthrough(input_pc, output_pc);
        }
        
    }

// Extract features of interest
    void FeatureExtractor::extractRegionOfInterest(const sensor_msgs::Image::ConstPtr& image,
                                                   const PointCloud::ConstPtr& pointcloud)
    {
        // Check if we have deduced the lidar ring count
        if (i_params.lidar_ring_count == 0)
        {
            // pcl::getMinMax3D only works on x,y,z
            for (const auto& p : pointcloud->points)
            {
                if (p.ring + 1 > i_params.lidar_ring_count)
                {
                    i_params.lidar_ring_count = p.ring + 1;
                }
            }
            lidar_frame_ = pointcloud->header.frame_id;
        }
        PointCloud::Ptr cloud_bounded(new PointCloud);
        distoffset_passthrough(pointcloud, cloud_bounded);

        // Publish the experimental region point cloud
        bounded_cloud_pub_.publish(cloud_bounded);

        if (flag == Optimise::Request::CAPTURE)
        {
            ROS_INFO("Processing sample");
            auto [corner_vectors, chessboard_normal] = locateChessboard(image);
            if (corner_vectors.size() == 0)
            {
                flag = Optimise::Request::READY;
                ROS_ERROR("Sample capture failed: can't detect chessboard in camera image");
                ROS_INFO("Ready to capture sample");
                return;
            }

            cam_lidar_calibration::OptimisationSample sample;
            num_samples++;
            sample.sample_num = num_samples;
            sample.camera_centre = corner_vectors[4];  // Centre of board
            corner_vectors.pop_back();
            sample.camera_corners = corner_vectors;
            sample.camera_normal = cv::Point3d(chessboard_normal);
            sample.pixeltometre = metreperpixel_cbdiag;

            // FIND THE MAX AND MIN POINTS IN EVERY RING CORRESPONDING TO THE BOARD
            auto [cloud_projected, lidar_normal] = extractBoard(cloud_bounded, sample);
            if (cloud_projected->points.size() == 0)
            {
                return;
            }
            sample.lidar_normal = lidar_normal;

            // First: Sort out the points in the point cloud according to their ring numbers
            std::vector<PointCloud> ring_pointclouds(i_params.lidar_ring_count);

            for (const auto& point : cloud_projected->points)
            {
                ring_pointclouds[point.ring].push_back(point);
            }

            // Second: Arrange points in every ring in descending order of y coordinate
            for (auto& ring : ring_pointclouds)
            {
                std::sort(ring.begin(), ring.end(), [](pcl::PointXYZIR p1, pcl::PointXYZIR p2) { return p1.y > p2.y; });
            }

            // Third: Find minimum and maximum points in a ring
            PointCloud::Ptr max_points(new PointCloud);
            PointCloud::Ptr min_points(new PointCloud);
            for (const auto& ring : ring_pointclouds)
            {
                if (ring.size() == 0)
                {
                    continue;
                }
                min_points->push_back(ring[ring.size() - 1]);
                max_points->push_back(ring[0]);
            }

            // Fit lines through minimum and maximum points
            auto [top_left, bottom_left] = findEdges(max_points);
            auto [top_right, bottom_right] = findEdges(min_points);

            if (top_left.values.empty() | top_right.values.empty()
            | bottom_left.values.empty() | bottom_right.values.empty()) {
                ROS_ERROR("RANSAC unsuccessful, discarding sample - Need more lidar points on board");
                pc_samples_.pop_back();
                num_samples--;
                flag = Optimise::Request::READY;
                ROS_INFO("Ready for capture\n");
                return;
            }

            // Get angles of targetboard
            cv::Mat top_left_vector = (cv::Mat_<double>(3,1) << top_left.values[3], top_left.values[4], top_left.values[5]);
            cv::Mat top_right_vector = (cv::Mat_<double>(3,1) << top_right.values[3], top_right.values[4], top_right.values[5]);
            cv::Mat bottom_left_vector = (cv::Mat_<double>(3,1) << bottom_left.values[3], bottom_left.values[4], bottom_left.values[5]);
            cv::Mat bottom_right_vector = (cv::Mat_<double>(3,1) << bottom_right.values[3], bottom_right.values[4], bottom_right.values[5]);
            double a0 = acos(top_left_vector.dot(top_right_vector))*180/M_PI;
            double a1 = acos(bottom_left_vector.dot(bottom_right_vector))*180/M_PI;
            double a2 = acos(top_left_vector.dot(bottom_left_vector))*180/M_PI;
            double a3 = acos(top_right_vector.dot(bottom_right_vector))*180/M_PI;
            sample.angles_0.push_back(a0);
            sample.angles_0.push_back(a1);
            sample.angles_1.push_back(a2);
            sample.angles_1.push_back(a3);

            // Find the corners
            // 3D Lines rarely intersect - lineWithLineIntersection has default threshold of 1e-4
            Eigen::Vector4f corner;
            pcl::lineWithLineIntersection(top_left, top_right, corner);
            cv::Point3d c0(corner[0], corner[1], corner[2]);
            pcl::lineWithLineIntersection(bottom_left, bottom_right, corner);
            cv::Point3d c1(corner[0], corner[1], corner[2]);
            pcl::lineWithLineIntersection(top_left, bottom_left, corner);
            cv::Point3d c2(corner[0], corner[1], corner[2]);
            pcl::lineWithLineIntersection(top_right, bottom_right, corner);
            cv::Point3d c3(corner[0], corner[1], corner[2]);
            // Add points in same order as the paper
            // Convert to mm
            sample.lidar_corners.push_back(c3 * 1000);
            sample.lidar_corners.push_back(c0 * 1000);
            sample.lidar_corners.push_back(c2 * 1000);
            sample.lidar_corners.push_back(c1 * 1000);

            for (const auto& p : sample.lidar_corners)
            {
                // Average the corners
                sample.lidar_centre.x += p.x / 4.0;
                sample.lidar_centre.y += p.y / 4.0;
                sample.lidar_centre.z += p.z / 4.0;
            }

            // Flip the lidar normal if it is in the wrong direction (mainly happens for rear facing cameras)
            double top_down_radius = sqrt(pow(sample.lidar_centre.x,2)+pow(sample.lidar_centre.y,2));
            double vector_dist = sqrt(pow(sample.lidar_centre.x + sample.lidar_normal.x,2) +
            	pow(sample.lidar_centre.y + sample.lidar_normal.y,2));
            if (vector_dist > top_down_radius) {
            	sample.lidar_normal.x = -sample.lidar_normal.x;
            	sample.lidar_normal.y = -sample.lidar_normal.y;
            	sample.lidar_normal.z = -sample.lidar_normal.z;
            }

            // Get line lengths for comparison with real board dimensions
            std::vector <double> lengths;
            lengths.push_back(sqrt(pow(c0.x - c3.x, 2) + pow(c0.y - c3.y, 2) + pow(c0.z - c3.z, 2)) * 1000);
            lengths.push_back(sqrt(pow(c0.x - c2.x, 2) + pow(c0.y - c2.y, 2) + pow(c0.z - c2.z, 2)) * 1000);
            lengths.push_back(sqrt(pow(c1.x - c3.x, 2) + pow(c1.y - c3.y, 2) + pow(c1.z - c3.z, 2)) * 1000);
            lengths.push_back(sqrt(pow(c1.x - c2.x, 2) + pow(c1.y - c2.y, 2) + pow(c1.z - c2.z, 2)) * 1000);
            std::sort(lengths.begin(), lengths.end());
            double w0 = lengths[0];
            double w1 = lengths[1];
            double h0 = lengths[2];
            double h1 = lengths[3];
            sample.widths.push_back(w0);
            sample.widths.push_back(w1);
            sample.heights.push_back(h0);
            sample.heights.push_back(h1);

            double gt_area = (double)i_params.board_dimensions.width/1000*(double)i_params.board_dimensions.height/1000;
            double b_area = (w0/1000*h0/1000)/2 + (w1/1000*h1/1000)/2;

            // Board dimension errors
            double w0_diff = abs(w0 - i_params.board_dimensions.width);
            double w1_diff = abs(w1 - i_params.board_dimensions.width);
            double h0_diff = abs(h0 - i_params.board_dimensions.height);
            double h1_diff = abs(h1 - i_params.board_dimensions.height);
            double be_dim_err = w0_diff + w1_diff + h0_diff + h1_diff;

            double distance = sqrt(pow(sample.lidar_centre.x/1000-0, 2) + pow(sample.lidar_centre.y/1000-0, 2) + pow(sample.lidar_centre.z/1000-0, 2));
            sample.distance_from_origin = distance;
            printf("\n--- Sample %d ---\n", num_samples);
            printf("Measured board has: dimensions = %dx%d mm; area = %6.5f m^2\n", i_params.board_dimensions.width, i_params.board_dimensions.height, gt_area);
            printf("Distance = %5.2f m\n", sample.distance_from_origin);
            printf("Board angles     = %5.2f,%5.2f,%5.2f,%5.2f degrees\n",a0, a1, a2, a3);
            printf("Board area       = %7.5f m^2 (%+4.5f m^2)\n", b_area, b_area-gt_area);
            printf("Board avg height = %6.2fmm (%+4.2fmm)\n", (h0+h1)/2, (h0+h1)/2-i_params.board_dimensions.height);
            printf("Board avg width  = %6.2fmm (%+4.2fmm)\n", (w0+w1)/2, (w0+w1)/2-i_params.board_dimensions.width);
            printf("Board dim        = %6.2f,%6.2f,%6.2f,%6.2f mm\n", w0, h0, h1, w1);
            printf("Board dim error  = %7.2f\n\n", be_dim_err);

            // If the lidar board dim is more than 10% of the measured, then reject sample
            if (abs(w0-i_params.board_dimensions.width) > i_params.board_dimensions.width*0.1 |
                abs(w1 - i_params.board_dimensions.width) > i_params.board_dimensions.width * 0.1 |
                abs(h0 - i_params.board_dimensions.height) > i_params.board_dimensions.height * 0.1 |
                abs(h1 - i_params.board_dimensions.height) > i_params.board_dimensions.height * 0.1) {
                ROS_ERROR("Plane fitting error, LiDAR board dimensions incorrect; discarding sample - try capturing again");
                pc_samples_.pop_back();
                num_samples--;
                flag = Optimise::Request::READY;
                ROS_INFO("Ready for capture\n");
                return;
            }

            ROS_INFO("Found line coefficients and outlined chessboard");
            publishBoardPointCloud();

            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

            // Save image 
            if(boost::filesystem::create_directory(newdatafolder))
            {   
                boost::filesystem::create_directory(newdatafolder + "/images");
                boost::filesystem::create_directory(newdatafolder + "/pcd");
                ROS_INFO_STREAM("Save data folder created at " << newdatafolder);
            } 
            std::string img_filepath = newdatafolder + "/images/pose" + std::to_string(num_samples)  + ".png" ;              
            std::string target_pcd_filepath = newdatafolder + "/pcd/pose" + std::to_string(num_samples)  + "_target.pcd" ;              
            std::string full_pcd_filepath = newdatafolder + "/pcd/pose" + std::to_string(num_samples)  + "_full.pcd" ;              
            
            ROS_ASSERT( cv::imwrite( img_filepath,  cv_ptr->image ) );   
            pcl::io::savePCDFileASCII (target_pcd_filepath, *cloud_bounded);
            pcl::io::savePCDFileASCII (full_pcd_filepath, *pointcloud);
            ROS_INFO_STREAM("Image and pcd file saved");


            if (num_samples == 1){
                // Check if save_dir has camera_info topic saved
                std::string pkg_path = ros::package::getPath("cam_lidar_calibration");

                std::ofstream camera_info_file;
                std::string camera_info_path = pkg_path + "/cfg/camera_info.yaml";
                ROS_INFO_STREAM("Camera_info saved at: " << camera_info_path);
                camera_info_file.open(camera_info_path, std::ios_base::out | std::ios_base::trunc);
                std::string dist_model = (i_params.fisheye_model) ? "fisheye": "non-fisheye";
                camera_info_file << "distortion_model: \"" << dist_model << "\"\n";
                camera_info_file << "width: " << i_params.image_size.first << "\n";
                camera_info_file << "height: " << i_params.image_size.second << "\n";
                camera_info_file << "D: [" << i_params.distcoeff.at<double>(0)
                                        << "," << i_params.distcoeff.at<double>(1)
                                        << "," << i_params.distcoeff.at<double>(2)
                                        << "," << i_params.distcoeff.at<double>(3) << "]\n";
                camera_info_file << "K: [" << i_params.cameramat.at<double>(0,0)
                                        << ",0.0"
                                        << "," << i_params.cameramat.at<double>(0,2)
                                        << ",0.0"
                                        << "," << i_params.cameramat.at<double>(1,1)
                                        << "," << i_params.cameramat.at<double>(1,2)
                                        << ",0.0,0.0" 
                                        << "," << i_params.cameramat.at<double>(2, 2) 
                                        << "]\n";
                camera_info_file.close();
            }

            // Push this sample to the optimiser
            optimiser_->samples.push_back(sample);
            flag = Optimise::Request::READY;  // Reset the capture flag
            ROS_INFO("Ready for capture\n");
        }  // if (flag == Optimise::Request::CAPTURE)
    }  // End of extractRegionOfInterest

// Get current date/time, format is YYYY-MM-DD-HH:mm:ss
    std::string FeatureExtractor::getDateTime()
    {
        auto time = std::time(nullptr);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%F_%T"); // ISO 8601 without timezone information.
        auto s = ss.str();
        std::replace(s.begin(), s.end(), ':', '-');
        return s;
    }

    // Function to find the octant in cartesian-polar transformation (for distance offset)
    int FeatureExtractor::find_octant(float x, float y, float z)
    {
        if (x >= 0 && y >= 0 && z >= 0)
            return 1;

        else if (x < 0 && y >= 0 && z >= 0)
            return 2;

        else if (x < 0 && y < 0 && z >= 0)
            return 3;

        else if (x >= 0 && y < 0 && z >= 0)
            return 4;

        else if (x >= 0 && y >= 0 && z < 0)
            return 5;

        else if (x < 0 && y >= 0 && z < 0)
            return 6;

        else if (x < 0 && y < 0 && z < 0)
            return 7;

        else if (x >= 0 && y < 0 && z < 0)
            return 8;
    }

}  // namespace cam_lidar_calibration
