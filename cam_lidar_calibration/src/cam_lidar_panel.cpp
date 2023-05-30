#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QTimer>

#include "cam_lidar_panel.h"
#include <cam_lidar_calibration/Optimise.h>

namespace cam_lidar_calibration
{
    CamLidarPanel::CamLidarPanel(QWidget* parent) : rviz::Panel(parent), action_client_("run_optimise", true)
    {
        optimise_client_ = nh_.serviceClient<cam_lidar_calibration::Optimise>("optimiser");
        action_client_.waitForServer();

        QVBoxLayout* main_layout = new QVBoxLayout;
        QHBoxLayout* button_layout = new QHBoxLayout;
        capture_button_ = new QPushButton("Capture sample");
        connect(capture_button_, SIGNAL(clicked()), this, SLOT(captureSample()));
        discard_button_ = new QPushButton("Discard last sample");
        connect(discard_button_, SIGNAL(clicked()), this, SLOT(discardSample()));
        optimise_button_ = new QPushButton("Optimise");
        optimise_button_->setEnabled(false);
        connect(optimise_button_, SIGNAL(clicked()), this, SLOT(optimise()));
        QTimer* timer = new QTimer;
        connect(timer, SIGNAL(timeout()), this, SLOT(updateResult()));
        timer->start(500);

        output_label_ = new QLabel("");

        button_layout->addWidget(capture_button_);
        button_layout->addWidget(discard_button_);
        auto button_group = new QGroupBox();
        button_group->setLayout(button_layout);
        main_layout->addWidget(button_group);
        main_layout->addWidget(optimise_button_);
        main_layout->addWidget(output_label_);

        setLayout(main_layout);
    }

    void CamLidarPanel::captureSample()
    {
        Optimise srv;
        srv.request.operation = Optimise::Request::CAPTURE;
        optimise_client_.call(srv);
        optimise_button_->setEnabled(true);
    }

    void CamLidarPanel::discardSample()
    {
        Optimise srv;
        srv.request.operation = Optimise::Request::DISCARD;
        optimise_client_.call(srv);
        optimise_button_->setEnabled(true);
    }

    void CamLidarPanel::optimise()
    {
        RunOptimiseGoal goal;
        action_client_.sendGoal(goal);
    }

    void CamLidarPanel::updateResult()
    {
        if (action_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        {
            capture_button_->setEnabled(true);
            discard_button_->setEnabled(true);
            optimise_button_->setEnabled(true);
            auto result = action_client_.getResult();
            auto t = result->transform.translation;
            auto r = result->transform.rotation;
            std::ostringstream os;
            os.precision(3);
            //    os << "Rotation - w: " << r.w << " x: " << r.x << " y: " << r.y << " z: " << r.z;
            //    os << "\nTranslation - x: " << t.x << " y: " << t.y << " z: " << t.z;
            os << "Finished - csv in cam_lidar_calibration/output";
            output_label_->setText(QString::fromStdString(os.str()));
        }
        if (action_client_.getState() == actionlib::SimpleClientGoalState::ACTIVE)
        {
            capture_button_->setEnabled(false);
            discard_button_->setEnabled(false);
            optimise_button_->setEnabled(false);
            std::string str = "Optimising...";
            // Make the ellipsis "bounce"
            int count = (output_label_->text().length() + 2) % 3;
            output_label_->setText(QString::fromStdString(str.substr(0, 11 + count)));
        }
    }

    // Save all configuration data from this panel to the given
    // Config object.  It is important here that you call save()
    // on the parent class so the class id and panel name get saved.
    void CamLidarPanel::save(rviz::Config config) const
    {
        rviz::Panel::save(config);
    }

    // Load all configuration data for this panel from the given Config object.
    void CamLidarPanel::load(const rviz::Config& config)
    {
        rviz::Panel::load(config);
    }

}  // end namespace cam_lidar_calibration

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(cam_lidar_calibration::CamLidarPanel, rviz::Panel)
