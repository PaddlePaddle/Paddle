#define _USE_MATH_DEFINES

#include "cam_lidar_calibration/optimiser.h"
#include "cam_lidar_calibration/point_xyzir.h"
#include <tf/transform_datatypes.h>

namespace cam_lidar_calibration
{

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

    void Optimiser::init_genes(RotationTranslation& p, const std::function<double(void)>& rnd01,
                               const RotationTranslation& initial_rot_trans, double angle_increment,
                               double translation_increment)
    {
        init_genes(p.rot, rnd01, initial_rot_trans.rot, angle_increment);

        std::vector<double> trans_vals;
        trans_vals.push_back(translation_increment);
        trans_vals.push_back(-translation_increment);
        int RandIndex = rand() % 2;
        p.x = initial_rot_trans.x + trans_vals.at(RandIndex) * rnd01();
        RandIndex = rand() % 2;
        p.y = initial_rot_trans.y + trans_vals.at(RandIndex) * rnd01();
        RandIndex = rand() % 2;
        p.z = initial_rot_trans.z + trans_vals.at(RandIndex) * rnd01();
    }

    double Optimiser::perpendicularCost(const Rotation& rot)
    {
        // We do all the alignment of features in the lidar frame
        // Eq (3) in the original baseline paper
        double cost = 0;
        for (const auto& sample : current_set_)
        {
            auto camera_normal_lidar_frame = rot * sample.camera_normal;
            auto perp = cv::Mat(sample.lidar_centre - sample.lidar_corners[0]).reshape(1);
            perp /= cv::norm(perp);
            cost += std::pow(perp.dot(camera_normal_lidar_frame), 2) / current_set_.size();
 
        }
        return cost;
    }

    double Optimiser::normalAlignmentCost(const Rotation& rot)
    {
        // Eq (4) in the original baseline paper
        // We do all the alignment of features in the lidar frame

        double cost;
        for (const auto& sample : current_set_)
        {
            auto camera_normal_lidar_frame = rot * sample.camera_normal;
            cost += cv::norm(camera_normal_lidar_frame - cv::Mat(sample.lidar_normal).reshape(1)) / current_set_.size();
        }
        return cost;
    }

    double Optimiser::reprojectionCost(const RotationTranslation& rot_trans)
    {
        // We do all the alignment of features in the lidar frame
        cv::Mat rvec = cv::Mat_<double>::zeros(3, 1);
        cv::Mat tvec = cv::Mat_<double>::zeros(3, 1);

        double cost = 0;
        for (auto& sample : current_set_)
        {

            std::vector<cv::Point3d> cam_centre_3d;
            std::vector<cv::Point3d> lidar_centre_3d;

            auto camera_centre_lidar_frame = cv::Point3d(rot_trans * sample.camera_centre);
            cam_centre_3d.push_back(camera_centre_lidar_frame);
            lidar_centre_3d.push_back(sample.lidar_centre);

            std::vector<cv::Point2d> cam, lidar;
            if (i_params_.fisheye_model)
            {
                cv::fisheye::projectPoints(cam_centre_3d, cam, rvec, tvec, i_params_.cameramat, i_params_.distcoeff);
                cv::fisheye::projectPoints(lidar_centre_3d, lidar, rvec, tvec, i_params_.cameramat, i_params_.distcoeff);
            }
            else
            {
                cv::projectPoints(cam_centre_3d, rvec, tvec, i_params_.cameramat, i_params_.distcoeff, cam);
                cv::projectPoints(lidar_centre_3d, rvec, tvec, i_params_.cameramat, i_params_.distcoeff, lidar);
            }

            double error = cv::norm(cam[0] - lidar[0])*sample.pixeltometre;

            if (error > cost)
            {
                cost = error;
            }
        }
        return cost;
    }

    double Optimiser::centreAlignmentCost(const RotationTranslation& rot_trans)
    {
        // Eq (6) and (7)
        // We do all the alignment of features in the lidar frame

        double abs_mean = 0;
        for (const auto& sample : current_set_)
        {
            auto camera_centre_lidar_frame = rot_trans * sample.camera_centre;
            abs_mean += cv::norm(camera_centre_lidar_frame - cv::Mat(sample.lidar_centre).reshape(1)) / current_set_.size();
        }
        double stddev = 0;
        for (const auto& sample : current_set_)
        {
            auto camera_centre_lidar_frame = rot_trans * sample.camera_centre;
            stddev += std::pow(cv::norm(camera_centre_lidar_frame - cv::Mat(sample.lidar_centre).reshape(1)) - abs_mean, 2) /
                    current_set_.size();
        }
        stddev = std::sqrt(stddev);

        return abs_mean/1000 + stddev/1000;
    }

    bool Optimiser::eval_solution(const RotationTranslation& p, RotationTranslationCost& c)
    {
        double perpendicular_cost = perpendicularCost(p.rot);
        double normal_align_cost = normalAlignmentCost(p.rot);

        double centre_align_cost = centreAlignmentCost(p);
        double repro_cost = reprojectionCost(p);
        c.objective2 = perpendicular_cost + normal_align_cost + centre_align_cost + repro_cost;

        return true;  // solution is accepted
    }

    RotationTranslation Optimiser::mutate(const RotationTranslation& X_base, const std::function<double(void)>& rnd01,
                                          const RotationTranslation& initial_rotation_translation,
                                          const double angle_increment, const double translation_increment,
                                          const double shrink_scale)
    {
        RotationTranslation X_new;

        bool in_range;
        do
        {
            in_range = true;
            X_new = X_base;
            float roll_inc = 0.2 * (rnd01() - rnd01()) * shrink_scale;
            X_new.rot.roll += roll_inc;
            in_range = in_range && (X_new.rot.roll >= (initial_rotation_translation.rot.roll - angle_increment) &&
                    X_new.rot.roll < (initial_rotation_translation.rot.roll + angle_increment));
            float pitch_inc = 0.2 * (rnd01() - rnd01()) * shrink_scale;
            X_new.rot.pitch += pitch_inc;
            in_range = in_range && (X_new.rot.pitch >= (initial_rotation_translation.rot.pitch - angle_increment) &&
                    X_new.rot.pitch < (initial_rotation_translation.rot.pitch + angle_increment));
            float yaw_inc = 0.2 * (rnd01() - rnd01()) * shrink_scale;
            X_new.rot.yaw += yaw_inc;
            in_range = in_range && (X_new.rot.yaw >= (initial_rotation_translation.rot.yaw - angle_increment) &&
                    X_new.rot.yaw < (initial_rotation_translation.rot.yaw + angle_increment));
            float x_inc = 0.2*1000 * (rnd01() - rnd01()) * shrink_scale;
            X_new.x += x_inc;
            in_range = in_range && (X_new.x >= (initial_rotation_translation.x - translation_increment) &&
                                    X_new.x < (initial_rotation_translation.x + translation_increment));
            float y_inc = 0.2*1000 * (rnd01() - rnd01()) * shrink_scale;
            X_new.y += y_inc;
            in_range = in_range && (X_new.y >= (initial_rotation_translation.y - translation_increment) &&
                                    X_new.y < (initial_rotation_translation.y + translation_increment));
            float z_inc = 0.2*1000 * (rnd01() - rnd01()) * shrink_scale;
            X_new.z += z_inc;
            in_range = in_range && (X_new.z >= (initial_rotation_translation.z - translation_increment) &&
                                    X_new.z < (initial_rotation_translation.z + translation_increment));
        } while (!in_range);
        return X_new;
    }

    RotationTranslation Optimiser::crossover(const RotationTranslation& X1, const RotationTranslation& X2,
                                             const std::function<double(void)>& rnd01)
    {
        RotationTranslation X_new;
        double r;
        r = rnd01();
        X_new.rot = crossover(X1.rot, X2.rot, rnd01);

        X_new.x = r * X1.x + (1.0 - r) * X2.x;
        r = rnd01();
        X_new.y = r * X1.y + (1.0 - r) * X2.y;
        r = rnd01();
        X_new.z = r * X1.z + (1.0 - r) * X2.z;
        return X_new;
    }

    double Optimiser::calculate_SO_total_fitness(const GA_Rot_Trans_t::thisChromosomeType& X)
    {
        // finalize the cost
        double final_cost = 0.0;
        final_cost += X.middle_costs.objective2;
        return final_cost;
    }

    // A function to show/store the results of each generation.
    void Optimiser::SO_report_generation(
            int generation_number, const EA::GenerationType<RotationTranslation, RotationTranslationCost>& last_generation,
            const RotationTranslation& best_genes)
    {
        best_rotation_translation_.rot = best_genes.rot;
        best_rotation_translation_.x = best_genes.x;
        best_rotation_translation_.y = best_genes.y;
        best_rotation_translation_.z = best_genes.z;
    }

    void Optimiser::init_genes(Rotation& p, const std::function<double(void)>& rnd01, const Rotation& initial_rotation,
                               double increment)
    {
        std::vector<double> pi_vals;
        pi_vals.push_back(increment);
        pi_vals.push_back(-increment);
        int RandIndex = rand() % 2;
        p.roll = initial_rotation.roll + pi_vals.at(RandIndex) * rnd01();
        RandIndex = rand() % 2;
        p.pitch = initial_rotation.pitch + pi_vals.at(RandIndex) * rnd01();
        RandIndex = rand() % 2;
        p.yaw = initial_rotation.yaw + pi_vals.at(RandIndex) * rnd01();
    }

    bool Optimiser::eval_solution(const Rotation& p, RotationCost& c)
    {
        c.objective1 = perpendicularCost(p) + normalAlignmentCost(p);

        return true;  // solution is accepted
    }

    Rotation Optimiser::mutate(const Rotation& X_base, const std::function<double(void)>& rnd01,
                               const Rotation& initial_rotation, const double angle_increment, double shrink_scale)
    {
        Rotation X_new;
        bool in_range;
        do
        {
            in_range = true;
            X_new = X_base;
            float roll_inc = 0.2 * (rnd01() - rnd01()) * shrink_scale;
            X_new.roll += roll_inc;
            in_range = in_range && (X_new.roll >= (initial_rotation.roll - angle_increment) &&
                                    X_new.roll < (initial_rotation.roll + angle_increment));
            float pitch_inc = 0.2 * (rnd01() - rnd01()) * shrink_scale;
            X_new.pitch += pitch_inc;
            in_range = in_range && (X_new.pitch >= (initial_rotation.pitch - angle_increment) &&
                                    X_new.pitch < (initial_rotation.pitch + angle_increment));
            float yaw_inc = 0.2 * (rnd01() - rnd01()) * shrink_scale;
            X_new.yaw += yaw_inc;
            in_range = in_range && (X_new.yaw >= (initial_rotation.yaw - angle_increment) &&
                                    X_new.yaw < (initial_rotation.yaw + angle_increment));
        } while (!in_range);
        return X_new;
    }

    Rotation Optimiser::crossover(const Rotation& X1, const Rotation& X2, const std::function<double(void)>& rnd01)
    {
        Rotation X_new;
        double r = rnd01();
        X_new.roll = r * X1.roll + (1.0 - r) * X2.roll;
        r = rnd01();
        X_new.pitch = r * X1.pitch + (1.0 - r) * X2.pitch;
        r = rnd01();
        X_new.yaw = r * X1.yaw + (1.0 - r) * X2.yaw;
        return X_new;
    }

    double Optimiser::calculate_SO_total_fitness(const GA_Rot_t::thisChromosomeType& X)
    {
        double final_cost = 0.0;  // finalize the cost
        final_cost += X.middle_costs.objective1;
        return final_cost;
    }

// A function to show/store the results of each generation.
    void Optimiser::SO_report_generation(int generation_number,
                                         const EA::GenerationType<Rotation, RotationCost>& last_generation,
                                         const Rotation& best_genes)
    {
        best_rotation_.roll = best_genes.roll;
        best_rotation_.pitch = best_genes.pitch;
        best_rotation_.yaw = best_genes.yaw;
    }

    void Optimiser::get_mean_stdev(std::vector<float>& input_vec, float& mean, float& stdev){
        float sum = std::accumulate(std::begin(input_vec), std::end(input_vec), 0.0);
        mean =  sum / input_vec.size();

        float accum = 0.0;
        std::for_each (std::begin(input_vec), std::end(input_vec), [&](const float d) {
            accum += (d - mean) * (d - mean);
        });

        stdev = sqrt(accum / (input_vec.size()-1));
    }

    // Generate combinations of size k from the total samples captured
    void Optimiser::generate_sets(int offset, int k, std::vector<OptimisationSample>& set, std::vector<OptimisationSample>& samples) {
        if (k == 0) {
            sets.push_back(set);
            return;
        }
        for (int i = offset; i <= samples.size() - k; ++i) {
            set.push_back(samples[i]);
            generate_sets(i+1, k-1, set, samples);
            set.pop_back();
        }
    }

    bool Optimiser::optimise(RotationTranslation& opt_result, std::vector<OptimisationSample>& set, cv::Mat& cameramat, cv::Mat& distcoeff)
    {
        // Update camera matrix/distortion coeff
        i_params_.cameramat = cameramat;
        i_params_.distcoeff = distcoeff;
        current_set_ = set;
        std::vector<float> b_dims;

        camera_centres_ = cv::Mat(current_set_.size(), 3, CV_64F);
        camera_normals_ = cv::Mat(current_set_.size(), 3, CV_64F);
        lidar_centres_ = cv::Mat(current_set_.size(), 3, CV_64F);
        lidar_normals_ = cv::Mat(current_set_.size(), 3, CV_64F);

        // Sum the errors of each sample and print the sample num
        for (auto& sample : current_set_)
        {
            float dim_sum = abs(sample.widths[0] - i_params_.board_dimensions.width)+abs(sample.widths[1] - i_params_.board_dimensions.width)+abs(sample.heights[0] - i_params_.board_dimensions.height)+abs(sample.heights[1] - i_params_.board_dimensions.height);
            b_dims.push_back(dim_sum);
        }

        // Insert vector elements into matrix to compute analytical euler angles by matrix operations
        int row = 0;
        for (auto& sample : current_set_)
        {
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

        cv::Mat NN = camera_normals_.t() * camera_normals_;
        cv::Mat NM = camera_normals_.t() * lidar_normals_;
        cv::Mat UNR = (NN.inv() * NM).t();  // Analytical rotation matrix for real data

        // This is not part of the process, just for verbose print statements
        float cn_cond_fro = cv::norm(camera_normals_, cv::NORM_L2) * cv::norm(camera_normals_.inv(), cv::NORM_L2);
        float ln_cond_fro = cv::norm(lidar_normals_, cv::NORM_L2) * cv::norm(lidar_normals_.inv(), cv::NORM_L2);

        float cond_max = (cn_cond_fro > ln_cond_fro) ? cn_cond_fro : ln_cond_fro;
        float b_avg = std::accumulate(std::begin(b_dims), std::end(b_dims), 0.0)/b_dims.size();

        // variability of quality (voq)
        float voq = cond_max + b_avg;
        printf("| voq: %7.3f ", voq);

        std::vector<double> euler = rotm2eul(UNR);
        EA::Chronometer timer;
        timer.tic();

        const Rotation initial_rotation{ euler[0], euler[1], euler[2] };
        double rotation_increment = M_PI / 8;
        namespace ph = std::placeholders;

        // Optimization for rotation alone
        GA_Rot_t ga_obj;
        ga_obj.problem_mode = EA::GA_MODE::SOGA;
        ga_obj.multi_threading = false;
        ga_obj.verbose = false;
        ga_obj.population = 200;
        ga_obj.generation_max = 1000;
        ga_obj.calculate_SO_total_fitness = [&](const GA_Rot_t::thisChromosomeType& X) -> double {
            return this->calculate_SO_total_fitness(X);
        };
        ga_obj.init_genes = [&, initial_rotation, rotation_increment](Rotation& p,
                                                                      const std::function<double(void)>& rnd01) -> void {
            this->init_genes(p, rnd01, initial_rotation, rotation_increment);
        };
        ga_obj.eval_solution = [&](const Rotation& r, RotationCost& c) -> bool { return this->eval_solution(r, c); };
        ga_obj.mutate = [&, initial_rotation, rotation_increment](
                const Rotation& X_base, const std::function<double(void)>& rnd01, double shrink_scale) -> Rotation {
            return this->mutate(X_base, rnd01, initial_rotation, rotation_increment, shrink_scale);
        };
        ga_obj.crossover = [&](const Rotation& X1, const Rotation& X2, const std::function<double(void)>& rnd01) {
            return this->crossover(X1, X2, rnd01);
        };
        ga_obj.SO_report_generation = [&](int generation_number,
                                          const EA::GenerationType<Rotation, RotationCost>& last_generation,
                                          const Rotation& best_genes) -> void {
            this->SO_report_generation(generation_number, last_generation, best_genes);
        };
        ga_obj.best_stall_max = 100;
        ga_obj.average_stall_max = 100;
        ga_obj.tol_stall_average = 1e-8;
        ga_obj.tol_stall_best = 1e-8;
        ga_obj.elite_count = 10;
        ga_obj.crossover_fraction = 0.8;
        ga_obj.mutation_rate = 0.2;
        ga_obj.best_stall_max = 10;
        ga_obj.elite_count = 10;
        ga_obj.solve();

        // Optimized rotation
        // Reset starting point of rotation genes
        tf::Matrix3x3 rot;
        rot.setRPY(best_rotation_.roll, best_rotation_.pitch, best_rotation_.yaw);
        cv::Mat tmp_rot = (cv::Mat_<double>(3, 3) << rot.getRow(0)[0], rot.getRow(0)[1], rot.getRow(0)[2], rot.getRow(1)[0],
                rot.getRow(1)[1], rot.getRow(1)[2], rot.getRow(2)[0], rot.getRow(2)[1], rot.getRow(2)[2]);
        // Analytical Translation
        cv::Mat cp_trans = tmp_rot * camera_centres_.t();
        cv::Mat trans_diff = lidar_centres_.t() - cp_trans;
        cv::Mat summed_diff;
        cv::reduce(trans_diff, summed_diff, 1, CV_REDUCE_SUM, CV_64F);
        summed_diff = summed_diff / trans_diff.cols;
        const RotationTranslation initial_rotation_translation{ best_rotation_, summed_diff.at<double>(0),
                                                                summed_diff.at<double>(1), summed_diff.at<double>(2) };

        rotation_increment = M_PI / 18;
        constexpr double translation_increment = 0.05*1000;
        // extrinsics stored the vector of extrinsic parameters in every iteration
        std::vector<std::vector<double>> extrinsics;
        // Joint optimization for Rotation and Translation (Perform this 10 times and take the average of the extrinsics)
        GA_Rot_Trans_t ga_rot_trans;
        ga_rot_trans.problem_mode = EA::GA_MODE::SOGA;
        ga_rot_trans.multi_threading = false;
        ga_rot_trans.verbose = false;
        ga_rot_trans.population = 200;
        ga_rot_trans.generation_max = 1000;
        ga_rot_trans.calculate_SO_total_fitness = [&](const GA_Rot_Trans_t::thisChromosomeType& X) -> double {
            return this->calculate_SO_total_fitness(X);
        };
        ga_rot_trans.init_genes = [&, initial_rotation_translation, rotation_increment, translation_increment](
                RotationTranslation& p, const std::function<double(void)>& rnd01) -> void {
            this->init_genes(p, rnd01, initial_rotation_translation, rotation_increment, translation_increment);
        };
        ga_rot_trans.eval_solution = [&](const RotationTranslation& rt, RotationTranslationCost& c) -> bool {
            return this->eval_solution(rt, c);
        };
        ga_rot_trans.mutate = [&, initial_rotation_translation, rotation_increment, translation_increment](
                const RotationTranslation& X_base, const std::function<double(void)>& rnd01,
                double shrink_scale) -> RotationTranslation {
            return this->mutate(X_base, rnd01, initial_rotation_translation, rotation_increment, translation_increment,
                                shrink_scale);
        };
        ga_rot_trans.crossover = [&](const RotationTranslation& X1, const RotationTranslation& X2,
                                        const std::function<double(void)>& rnd01) { return this->crossover(X1, X2, rnd01); };
        ga_rot_trans.SO_report_generation =
                [&](int generation_number,
                    const EA::GenerationType<RotationTranslation, RotationTranslationCost>& last_generation,
                    const RotationTranslation& best_genes) -> void {
                    this->SO_report_generation(generation_number, last_generation, best_genes);
                };
        ga_rot_trans.best_stall_max = 100;
        ga_rot_trans.average_stall_max = 100;
        ga_rot_trans.tol_stall_average = 1e-8;
        ga_rot_trans.tol_stall_best = 1e-8;
        ga_rot_trans.elite_count = 10;
        ga_rot_trans.crossover_fraction = 0.8;
        ga_rot_trans.mutation_rate = 0.2;
        ga_rot_trans.best_stall_max = 10;
        ga_rot_trans.elite_count = 10;
        ga_rot_trans.solve();

        opt_result.rot.roll = best_rotation_translation_.rot.roll;
        opt_result.rot.pitch = best_rotation_translation_.rot.pitch;
        opt_result.rot.yaw = best_rotation_translation_.rot.yaw;
        opt_result.x = best_rotation_translation_.x;
        opt_result.y = best_rotation_translation_.y;
        opt_result.z = best_rotation_translation_.z;

        printf("| % 05.3f,% 05.3f,% 05.3f,% 05.3f,% 05.3f,% 05.3f ", opt_result.rot.roll,opt_result.rot.pitch,opt_result.rot.yaw,opt_result.x / 1000.0,opt_result.y / 1000.0,opt_result.z / 1000.0);
        return true;
    }

// Function converts rotation matrix to corresponding euler angles
    std::vector<double> rotm2eul(cv::Mat mat)
    {
        std::vector<double> euler(3);
        euler[0] = atan2(mat.at<double>(2, 1), mat.at<double>(2, 2));  // rotation about x axis: roll
        euler[1] = atan2(-mat.at<double>(2, 0),
                         sqrt(mat.at<double>(2, 1) * mat.at<double>(2, 1) + mat.at<double>(2, 2) * mat.at<double>(2, 2)));
        euler[2] = atan2(mat.at<double>(1, 0), mat.at<double>(0, 0));  // rotation about z axis: yaw
        return euler;
    }

    Optimiser::Optimiser(const initial_parameters_t& params) : i_params_(params)
    {}
}  // namespace cam_lidar_calibration
