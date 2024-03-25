#ifndef optimiser_h_
#define optimiser_h_

// For writing to CSV
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>
#include<opencv2/core/core_c.h>

#include <ros/ros.h>

#include "cam_lidar_calibration/load_params.h"
#include "cam_lidar_calibration/openga.h"

namespace cam_lidar_calibration
{
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

            cv::Mat R_x = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll));
            // Calculate rotation about y axis
            cv::Mat R_y = (Mat_<double>(3, 3) << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch));

            // Calculate rotation about z axis
            cv::Mat R_z = (Mat_<double>(3, 3) << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);

            return R_z * R_y * R_x;
        }
    };

    cv::Mat operator*(const Rotation& lhs, const cv::Point3d& rhs);

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

    cv::Mat operator*(const RotationTranslation& lhs, const cv::Point3d& rhs);

    struct RotationCost  // equivalent to y in matlab
    {
        double objective1;  // This is where the results of simulation is stored but not yet finalized.
    };

    struct RotationTranslationCost  // equivalent to y in matlab
    {
        double objective2;  // This is where the results of simulation is stored but not yet finalized.
    };

    struct OptimisationSample
    {
        cv::Point3d camera_centre{ 0, 0, 0 };
        cv::Point3d camera_normal{ 0, 0, 0 };
        std::vector<cv::Point3d> camera_corners;
        cv::Point3d lidar_centre{ 0, 0, 0 };
        cv::Point3d lidar_normal{ 0, 0, 0 };
        std::vector<cv::Point3d> lidar_corners;
        std::vector<double> angles_0; // Currently unused - only for print statements on capture
        std::vector<double> angles_1; // Currently unused - only for print statements on capture
        std::vector<double> widths;
        std::vector<double> heights;
        float distance_from_origin; // Currently unused - only for print statements on capture
        double pixeltometre;
        int sample_num;
    };

    struct SetAssess
    {
        float voq;
        std::vector<OptimisationSample> set;
    };

    typedef EA::Genetic<Rotation, RotationCost> GA_Rot_t;
    typedef EA::Genetic<RotationTranslation, RotationTranslationCost> GA_Rot_Trans_t;

    class Optimiser
    {
    public:
        Optimiser(const initial_parameters_t& params);
        ~Optimiser() = default;

        bool optimise(RotationTranslation& opt_result, std::vector<OptimisationSample>& set, cv::Mat& cameramat, cv::Mat& distcoeff);
        std::vector<OptimisationSample> samples;
        std::vector<OptimisationSample> current_set_;
        std::vector<std::vector<OptimisationSample>> sets;
        std::vector<std::vector<OptimisationSample>> top_sets;
        std::map<int, float> top_idxqos;
        cv::Mat camera_centres_, camera_normals_, lidar_centres_, lidar_normals_;
        void generate_sets(int offset, int k, std::vector<OptimisationSample>& set, std::vector<OptimisationSample>& samples);

        // Rotation only
        void SO_report_generation(int generation_number, const EA::GenerationType<Rotation, RotationCost>& last_generation,
                                  const Rotation& best_genes);
        double calculate_SO_total_fitness(const GA_Rot_t::thisChromosomeType& X);
        Rotation crossover(const Rotation& X1, const Rotation& X2, const std::function<double(void)>& rnd01);
        Rotation mutate(const Rotation& X_base, const std::function<double(void)>& rnd01, const Rotation& initial_rotation,
                        const double angle_increment, const double shrink_scale);
        bool eval_solution(const Rotation& p, RotationCost& c);
        void init_genes(Rotation& p, const std::function<double(void)>& rnd01, const Rotation& initial_rotation,
                        double increment);

        // Rotation and translation
        void SO_report_generation(int generation_number,
                                  const EA::GenerationType<RotationTranslation, RotationTranslationCost>& last_generation,
                                  const RotationTranslation& best_genes);
        double calculate_SO_total_fitness(const GA_Rot_Trans_t::thisChromosomeType& X);
        RotationTranslation crossover(const RotationTranslation& X1, const RotationTranslation& X2,
                                      const std::function<double(void)>& rnd01);
        RotationTranslation mutate(const RotationTranslation& X_base, const std::function<double(void)>& rnd01,
                                   const RotationTranslation& initial_rotation_translation, const double angle_increment,
                                   const double translation_increment, const double shrink_scale);
        bool eval_solution(const RotationTranslation& p, RotationTranslationCost& c);
        void init_genes(RotationTranslation& p, const std::function<double(void)>& rnd01,
                        const RotationTranslation& initial_rotation_translation, double angle_increment,
                        double translation_increment);

    private:
        double perpendicularCost(const Rotation& rot);
        double normalAlignmentCost(const Rotation& rot);
        double reprojectionCost(const RotationTranslation& rot_trans);
        double centreAlignmentCost(const RotationTranslation& rot_trans);
        std::vector<double> analytical_euler(std::vector<OptimisationSample>& set,
                                             cv::Mat& camera_centres_,
                                             cv::Mat& camera_normals_,
                                             cv::Mat& lidar_centres_,
                                             cv::Mat& lidar_normals_);
        void get_mean_stdev(std::vector<float>& input, float& mean, float& stdev);

        Rotation best_rotation_;
        RotationTranslation best_rotation_translation_;
        initial_parameters_t i_params_;
    };

    std::vector<double> rotm2eul(cv::Mat);
    int det( int matrix[3][3], int n);

}  // namespace cam_lidar_calibration

#endif
