#ifndef cam_lidar_panel_h_
#define cam_lidar_panel_h_

#ifndef Q_MOC_RUN
#include <ros/ros.h>

#include <rviz/panel.h>
#endif

#include <QLabel>
#include <QPushButton>

#include <actionlib/client/simple_action_client.h>
#include <cam_lidar_calibration/RunOptimiseAction.h>

using ActionClient = actionlib::SimpleActionClient<cam_lidar_calibration::RunOptimiseAction>;

namespace cam_lidar_calibration
{
    class CamLidarPanel : public rviz::Panel
    {
        // This class uses Qt slots and is a subclass of QObject, so it needs
        // the Q_OBJECT macro.
    Q_OBJECT
    public:
        // QWidget subclass constructors usually take a parent widget
        // parameter (which usually defaults to 0).  At the same time,
        // pluginlib::ClassLoader creates instances by calling the default
        // constructor (with no arguments).  Taking the parameter and giving
        // a default of 0 lets the default constructor work and also lets
        // someone using the class for something else to pass in a parent
        // widget as they normally would with Qt.
        CamLidarPanel(QWidget* parent = 0);

        // Now we declare overrides of rviz::Panel functions for saving and
        // loading data from the config file.
        void load(const rviz::Config& config) override;
        void save(rviz::Config config) const override;

    public Q_SLOTS:

    protected Q_SLOTS:
        void captureSample();
        void discardSample();
        void optimise();
        void updateResult();

    protected:
        // The ROS node handle.
        ros::NodeHandle nh_;
        ros::NodeHandle private_nh_;
        bool import_samples_;
        ros::ServiceClient optimise_client_;
        ActionClient action_client_;

        QLabel* output_label_;
        QPushButton* capture_button_;
        QPushButton* discard_button_;
        QPushButton* optimise_button_;
    };

}  // end namespace cam_lidar_calibration

#endif  // cam_lidar_panel_h_
