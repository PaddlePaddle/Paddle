# 标定功能包介绍

包含单目、双目、雷达、IMU的多种方法

---

 **`calibration_ros`** 目录下包含以下**功能包**：

+ `usb_cam` 启动usb摄像头功能包,提供`cameraInfo与image_raw`两个话题

+ **`cam_calibration_pkg`** python脚本程序,包含单目,双目的拍照,标定功能

+ **`image_pipeline`** ros官网提供的关于相机图像的操作,包含`camera_calibration`功能包,

  只标定可只单独编译`camera_calibration`,标定需要提供同时`cameraInfo与image_raw`两个话题

+ **`imu_utils`与`code_utils`** 标定imu的零偏、方差工具,`code_utils`为依赖,需要先编译
  
  imu标定数据包一般录制静止时2个小时左右

+ **`cam_lidar_calibration`** 相机与雷达联合标定

+ **`kalibr`** 提供多目标定,imu与相机联合标定,多目标定相机时播放数据频率推荐4Hz,不宜太快

  编译`kalibr`时间比较久,电脑性能差半个小时起步,慢慢等待,电脑差编译`-j2`就行

  `kalibr`中的`kalibr_create_target_pdf`可以用来**制作标定板**

---

单独编译使用以下代码

~~~sh
catkin build camera_calibration -DCMAKE_BUILD_TYPE=Release -j4
~~~

`camera_calibration`换`image_pipeline、usb_cam、code_utils、imu_utils、kalibr、cam_lidar_calibration`即可


---


## 标定板与AprilTags二维码制作

1. `kalibr`中的`kalibr_create_target_pdf`可以用来**制作标定板**
~~~sh
# 棋盘格checkerboard
rosrun kalibr kalibr_create_target_pdf --type checkerboard --nx 6 --ny 6 --csx 0.04 --csy 0.04
# 四月格apriltag
rosrun kalibr kalibr_create_target_pdf --type apriltag --nx 6 --ny 6 --tsize 0.02 --tspace 0.5
~~~

2. `calibration-checkerboard` 官网生成棋盘格

[calibration-checkerboard官网](https://markhedleyjones.com/projects/calibration-checkerboard-collection)

![](img/2023-01-03_16-15.png)


设置好 `Page size与Checker size` 点击箭头指向的文件下载为`svg`格式,`svg`选择打印生成`pdf`文件


3. `openMV生成AprilTags` 二维码

`AprilTags`二维码可用于定位,[AprilTags官网](https://april.eecs.umich.edu/software/apriltag)

+ 可以使用`openMV生成AprilTags`[openMV安装地址](https://openmv.io/pages/download) 下载`ubuntu`版本

`AprilTag`是一个视觉基准系统，类似于二维码，但降低了复杂度以满足实时性需求，`AprilTag`具有很小的数据有效载荷（4到12位编码）。

相对于二维码，`AprilTag`可以在更远的距离，非常低的分辨率，不均匀照明，奇怪的旋转或藏在另一个杂乱图像的角落时也可以被自动检测和定位。

它们的设计具有很高的定位精度，我们可以计算相机相对于`AprilTag的精确6自由度位姿`信息。


+ 

---


+

# 1 简介

提供的标定数据分为ros包，图片(png,jpg等),传感器实时数据三种

传感器联合标定前需要单独标定各个传感器,激光一般默认无需标定


## 1.1 单目相机标定

图片，使用 `cam_calibration_pkg` 的程序进行标定

ros包，提供`camerainfo`使用`camera_calibration cameracalibrator.py`标定

传感器实时数据：都可以

## 1.2 多目相机标定

先单独标定每个相机,然后联合标定,这里只讲多目标定

1. `cam_calibration_pkg`

`cam_calibration_pkg`中`stereo_calibration_node.py`提供双目标定,本地提供图片数据

2. `ros`中`camera_calibration`包双目标定

提供左右目得`相机info`与`图像/image_raw`共4个话题

3. `kalibr`

`kalibr`中`kalibr_calibrate_cameras`功能包可以进行多目标定

只需要提供离线ros数据包,相机话题频率4Hz,不要太快

## 1.3 IMU与相机联合标定

先分别对imu与相机内参进行标定，再联合标定

imu单独标定使用[imu_utils](https://github.com/gaowenliang/imu_utils)工具

先录制包相机频率建议4Hz,然后离线标定,使用`kalibr`中得功能包`kalibr_calibrate_imu_camera`

## 1.4 激光与相机联合标定

激光一般不用单独标定,相机先单独内参标定，再联合标定

使用`cam_lidar_calibration中run_optimiser.launch`


---


# 2 相机标定

[相机标定原理](https://mp.weixin.qq.com/s?__biz=MzU0NjgzMDIxMQ==&mid=2247591782&idx=3&sn=511cd9483b91ed13c6d58f36f16446cd&chksm=fb54858acc230c9c2d870a0edb6973d7d8107d135b94086520d194ebc8f9b92d7c346a489cca&scene=27)

## 2.1 自写cam_calibration_pkg

手写的python版本的标定程序，包含拍照，单目标定

+ 运行前，先生效环境变量
~~~sh
soure devel/setup.bash
~~~

+ python 环境和ws_path
~~~python
#!/home/lin/software/miniconda3/envs/yolov5/bin/python
#！/usr/bin/python3
#-*- coding:utf-8 -*-

# 功能包的路径，建议直接绝对路径
ws_path = "/home/lin/ros_code/calibration_ws/src/calibration_ros/cam_calibration_pkg"
~~~

+ 单目标定

1. 拍摄棋盘格
~~~sh
rosrun cam_calibration_pkg mono_take_photo.py 
~~~

拍摄图片保存在`./data/mono/`目录下

2. 标定
~~~sh
rosrun cam_calibration_pkg mono_calibration_node.py 
~~~

结果保存在`./result/mono_calib.txt`文件中，矫正图像在`./undistort`目录下

+ 双目标定

1. 拍摄棋盘格
~~~sh
rosrun cam_calibration_pkg stereo_take_photo.py 
~~~

拍摄图片保存在`./data/left/`与`./data/right/`目录下

2. 标定
~~~sh
rosrun cam_calibration_pkg stereo_calibration_node.py 
~~~

结果保存在`./result/stereo_calib.txt`文件中，矫正图像在`./stero_rectify`目录下


---

## 2.2 ros标定相机`image_pipeline`中`camera_calibration`包


使用该功能包需要同时提供`相机info和图像img_raw`两个话题

~~~python
# 单目
image:=/camera/image_raw 
camera:=/camera_info

# 双目
right:=/my_stereo/right/image_raw 
left:=/my_stereo/left/image_raw 
left_camera:=/my_stereo/left_info 
right_camera:=/my_stereo/right_info
~~~

### 2.2.1 usb_cam

打开usb摄像头，需要修改launch文件中得摄像头序号


### 2.2.2 image_pipeline

[image_pipeline](https://github.com/ros-perception/image_pipeline)该功能包里面的`camera_calibration`包行ros官方提供的单目，双目在线标定

使用时需要指定`CameraInfo`

+ 下载编译`camera_calibration`

ubunt20.04下载noetic;ubuntu18.04下载melodic

~~~sh
cd calibration_ws/src
https://github.com/ros-perception/image_pipeline.git -b noetic
cd ..
catkin build camera_calibration
~~~

+ 单目标定

1. 启动usb_cam，该功能包包含`cameraInfo`信息，不需要自己手写了

修改launch中usb的序号`ls /dev/video*`查看

~~~sh
 roslaunch usb_cam usb_cam-test.launch 
~~~

2. 启动单目标定

~~~sh
rosrun camera_calibration cameracalibrator.py --size 6x4 --square 0.95 image:=/usb_cam/image_raw camera:=/usb_cam/camera_info --no-service-check
~~~

~~~sh
--size # 内角点数目
--square 小方格边长, # 单位m
image:=/usb_cam/image_raw    # 图像话题名
camera:=/usb_cam/camera_info  # 相机话题名字
~~~

3. 启动上面代码后如图所示：

可以选择camera type ：针孔和鱼眼

右边有3个按钮，刚开始无法点击(灰色），

需要检测到棋盘格，并收集45张图片，可以再终端查看进程

45张收集完成了，就可以点击calibrate(绿色)了

开始标定，会卡一会，等待计算，完成后save按钮变绿色，点击，结果保存到`/tmp/calibrationdata.tar.gz`

![](img/2022-12-20-13-59-54.png)

---

+ [双目标定](http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration)

修改对面的话题名字，size和square,一般采用ros包发布数据

--approximate 选项允许相机校准器处理不具有完全相同时间戳的图像,当前设置为 0.1 秒。

在这种情况下，只要时间戳差异小于 0.1 秒，校准器就可以正常运行

~~~sh
rosrun camera_calibration cameracalibrator.py --approximate 0.1 --size 8x6 --square 0.108 right:=/my_stereo/right/image_raw left:=/my_stereo/left/image_raw left_camera:=/my_stereo/left right_camera:=/my_stereo/right
~~~

侧边栏将显示棋盘正方形的测量精度和尺寸,epi精度(单位像素)0.16,dim为尺寸

![](img/230103-1447.png)


通常，低于 0.25 像素的对极误差被认为是可以接受的，低于 0.1 的极佳。

---

## 2.3 kalibr标定相机

[csdn参考链接](https://blog.csdn.net/qq_34570910/article/details/103566490)
[官方教程不同ubuntu下安装教程](https://github.com/ethz-asl/kalibr/wiki/installation)

使用kalibr一般先录制ros包后，离线标定

[官方提供数据包](https://github.com/ethz-asl/kalibr/wiki/downloads)

kalibr在处理标定数据的时候要求频率不能太高，一般为4Hz，我们可以使用如下命令来更改topic的频率

1. 改变topic频率

~~~sh
# rosrun topic_tools throttle messages 输入话题 频率 输出话题
rosrun topic_tools throttle messages <intopic> <msgs_per_sec> [outtopic]
~~~

~~~sh
rosrun topic_tools throttle messages /camera/color/image_raw 4.0 /outtopic
~~~

2. 录制topic

测试采用[官方提供数据包](https://github.com/ethz-asl/kalibr/wiki/downloads),实际可以采用rosbag录制数据包

~~~sh
rosbag record -O multicameras_calibration /infra_left /infra_right /color
~~~

3. 下载编译kalibr的依赖

+ 所有版本的 Ubuntu 通用的一般要求如下：

`libopencv-dev ,libeigen3-dev`两个依赖是源码编译得这里就不安装了
~~~sh
sudo apt-get install -y  libboost-all-dev libsuitesparse-dev doxygen libpoco-dev libtbb-dev libblas-dev liblapack-dev libv4l-dev
~~~

+ 然后由于不同的Python版本，你需要安装如下：
~~~sh
# Ubuntu 16.04
sudo apt-get install -y python2.7-dev python-pip python-scipy python-matplotlib ipython python-wxgtk3.0 python-tk python-igraph

# Ubuntu 18.04
sudo apt-get install -y python3-dev python-pip python-scipy python-matplotlib ipython python-wxgtk4.0 python-tk python-igraph

# Ubuntu 20.04
sudo apt-get install -y python3-dev python3-pip python3-scipy python3-matplotlib ipython3 python3-wxgtk4.0 python3-tk python3-igraph
~~~

4. 编译

[不同ubuntu下安装教程](https://github.com/ethz-asl/kalibr/wiki/installation)

+ ethz-asl/kalibr.git版本

Ubuntu 16.04-Ubuntu 20.04都适用

~~~sh
cd ~/kalibr_ws/src
git clone https://github.com/ethz-asl/kalibr.git
cd ~/kalibr_ws/
# 单独编译
catkin build kalibr -DCMAKE_BUILD_TYPE=Release -j4
~~~

5. 标定

+ 播放数据`imu_april.bag`

下载`IMU-CAM`数据包

`imu_april.bag`包含左右相机和imu话题

图像的参数：height: 480,width: 752

~~~sh
rosbag play imu_april.bag

# 该数据集中的话题有：
/cam0/image_raw
/cam1/image_raw
/clock
/imu0
/rosout
/rosout_agg
~~~

+ 启动多目相机标定
~~~sh
rosrun kalibr kalibr_calibrate_cameras --target data/kalibr_sample/static/april_6x6.yaml --bag data/kalibr_sample/static/static.bag --models pinhole-equi pinhole-equi omni-radtan --topics /cam0/image_raw /cam1/image_raw /cam2/image_raw --bag-from-to 2 4
~~~

+ 上面`imu_april.bag`数据包为双目

~~~sh
rosrun kalibr kalibr_calibrate_cameras --target april_6x6.yaml --bag imu_april.bag --models pinhole-radtan pinhole-radtan --topics /cam0/image_raw /cam1/image_raw 
~~~

参数说明
~~~sh
--bag:标定数据的名称
--topics:左右目相机的topic
--models:左右目相机模型
# pinhole-radtan: 最常见的针孔模型+布朗畸变模型(切径向畸变), 适用于大多数的角度小于120的相机, 其中畸变参数包含了径向畸变k1,k2和切向畸变p1,p2; 如果相机的畸变情况不是很严重,这个模型基本都可以; 比如我的DFOV为150的相机, 也可以用这个且去畸变效果很好;
# pinhole-equi:针孔模型＋等距畸变模型，也就是ＫＢ模型所需要选择的类型，该模型的使用范围也很广，大部分的鱼眼镜头也可以，注意8参数的ＫＢ模型的畸变参数为k1,k2,k3,k4，虽然也是四个数，但与前一个模型不同的是，这里只有径向畸变的参数，而没有切向畸变tangential distortion，投影时对应的公式也不同；同时这也是opencv中cv::fisheye使用的模型；
# omni-radtan全向相机模型+切径向畸变
--target:标定板参数配置文件
--approx-sync 时间同步容忍度
~~~

[--models参数官方说明](https://github.com/ethz-asl/kalibr/wiki/supported-models)

[--target参数官网说明](https://github.com/ethz-asl/kalibr/wiki/calibration-targets)

四月格`aprilgrid.yaml`
~~~
target_type: 'aprilgrid' # gridtype
tagCols: 6               # number of apriltags
tagRows: 6               # number of apriltags
tagSize: 0.088           # size of apriltag, edge to edge [m]
tagSpacing: 0.3          # ratio of space between tags to tagSize
                         # example: tagSize=2m, spacing=0.5m --> tagSpacing=0.25[-]
                         # 实际上就是小黑块与大黑块的边长之比 example: tagSize=2m, spacing=0.5m --> tagSpacing=0.25
~~~

棋盘格`checkerboard.yaml`
~~~
target_type: 'checkerboard' #gridtype
targetCols: 6               #number of internal chessboard corners
targetRows: 7               #number of internal chessboard corners
rowSpacingMeters: 0.06      #size of one chessboard square [m]
colSpacingMeters: 0.06      #size of one chessboard square [m]
~~~

--bag ros包

--topics 话题名,这里是3个话题名字,和--models对应也是3个

+ 校准将产生以下输出：

report-cam-％BAGNAME％.pdf：以PDF格式报告。包含所有用于文档的图。
results-cam-％BAGNAME％.txt：结果摘要为文本文件。
camchain-％BAGNAME％.yaml：结果为YAML格式。该文件可用作相机imu校准器的输入。

包含：
1.相机的重投影误差，IMU的误差（加速度和陀螺仪）可以作为先验误差来影响数据融合的定权问题
2.相机和IMU各自的标定参数，2个.yaml文件给的
3.IMU与相机之间的相对位姿标定（正反旋转矩阵）cam1 to imu0也有

~~~sh
T_ci:  (imu0 to cam0): 
[[ 0.01680206  0.99985864 -0.00062288  0.06847911]
 [-0.99985871  0.01680236  0.00048881 -0.01472898]
 [ 0.00049921  0.00061458  0.99999969 -0.00376988]
 [ 0.          0.          0.          1.        ]]

T_ic:  (cam0 to imu0): 
[[ 0.01680206 -0.99985871  0.00049921 -0.0158756 ]
 [ 0.99985864  0.01680236  0.00061458 -0.06821963]
 [-0.00062288  0.00048881  0.99999969  0.00381973]
 [ 0.          0.          0.          1.        ]]

相机之间的位姿变换标定（基线baseline）：
Baseline (cam0 to cam1): 
[[ 0.99999877  0.00118911 -0.00102243 -0.1101676 ]
 [-0.00118838  0.99999904  0.00071255 -0.00032166]
 [ 0.00102327 -0.00071134  0.99999922  0.00012079]
 [ 0.          0.          0.          1.        ]]
baseline norm:  0.110168134052 [m]
~~~

---



+ ori-drs/kalibr.git版本

这里只下载了noetic-devel

20.04下载noetic-devel分支
~~~sh
git clone https://github.com/ori-drs/kalibr.git --branch noetic-devel
~~~

用于rosdep安装几乎所有必需的依赖项

~~~sh
rosdep install --from-paths ./ -iry
~~~

然后安装两个缺少的运行时依赖项：
`sudo apt install python3-wxgtk4.0 python3-igraph`

---

# 3 IMU与相机联合标定

## 3.1 imu_utils工具标定IMU

imu单独标定使用[imu_utils](https://github.com/gaowenliang/imu_utils)工具

imu_utils用于分析 IMU 性能的 ROS 封装工具。Allan 方差工具的 C++ 版本。

实际上，只需分析 IMU 数据的 Allan Variance。**在 IMU 静止时收集数据，持续时间为两个小时**。

+ imu_utils工具安装与使用

依赖
~~~sh
sudo apt-get install libdw-dev
~~~

`code_utils`依赖

[编译报错参考链接](https://blog.csdn.net/learning_tortosie/article/details/102415313)

~~~sh
cd imu_ws/src
git clone https://github.com/gaowenliang/code_utils.git
cd ..
catkin build code_utils
~~~

`imu_utils`下载编译

~~~sh
cd imu_ws/src
git clone https://github.com/gaowenliang/imu_utils.git
cd ..
catkin build imu_utils
~~~

在IMU静止时收集数据，持续两小时

[官方提供数据包](https://github.com/ethz-asl/kalibr/wiki/downloads)

~~~sh
# 播放数据
rosbag play -r 200 imu_A3.bag

# 修改A3.launch文件
roslaunch imu_utils A3.launch
~~~

## 3.2 相机标定

参考2小节的相机标定

## 3.3 IMU与相机联合标定

使用`kalibr`中`kalibr_calibrate_imu_camera`功能包进行标定

[csdn参考链接](https://blog.csdn.net/qq_34570910/article/details/103566490)

[官方提供数据包](https://github.com/ethz-asl/kalibr/wiki/downloads)

~~~sh
rosrun kalibr kalibr_calibrate_imu_camera --target april_6x6.yaml --cam camchain.yaml --imu imu_adis16448.yaml --bag dynamic.bag --bag-from-to 5 45
~~~

+ 参数说明
1. `--cam camchain.yaml`
~~~sh
cam0:
  camera_model: pinhole
  intrinsics: [461.629, 460.152, 362.680, 246.049]
  distortion_model: radtan
  distortion_coeffs: [-0.27695497, 0.06712482, 0.00087538, 0.00011556]
  T_cam_imu:
  - [0.01779318, 0.99967549,-0.01822936, 0.07008565]
  - [-0.9998017, 0.01795239, 0.00860714,-0.01771023]
  - [0.00893160, 0.01807260, 0.99979678, 0.00399246]
  - [0.0, 0.0, 0.0, 1.0]
  timeshift_cam_imu: -8.121e-05
  rostopic: /cam0/image_raw
  resolution: [752, 480]
cam1:
  camera_model: omni
  intrinsics: [0.80065662, 833.006, 830.345, 373.850, 253.749]
  distortion_model: radtan
  distortion_coeffs: [-0.33518750, 0.13211436, 0.00055967, 0.00057686]
  T_cn_cnm1:
  - [ 0.99998854, 0.00216014, 0.00427195,-0.11003785]
  - [-0.00221074, 0.99992702, 0.01187697, 0.00045792]
  - [-0.00424598,-0.01188627, 0.99992034,-0.00064487]
  - [0.0, 0.0, 0.0, 1.0]
  T_cam_imu:
  - [ 0.01567142, 0.99978002,-0.01393948,-0.03997419]
  - [-0.99966203, 0.01595569, 0.02052137,-0.01735854]
  - [ 0.02073927, 0.01361317, 0.99969223, 0.00326019]
  - [0.0, 0.0, 0.0, 1.0]
  timeshift_cam_imu: -8.681e-05
  rostopic: /cam1/image_raw
  resolution: [752, 480]
~~~

~~~sh
CAMERA_MODEL
camera_model（pinhole / omni）（针孔、全向）

intrinsics
包含给定投影类型的内部参数的向量。要素如下：
pinhole：[fu fv pu pv]
omn​​i：[xi fu fv pu pv]
ds：[xi alpha fu fv pu pv]
eucm：[alpha beta fu fv pu pv]

distortion_model
distortion_model（radtan /equidistant）

distortion_coeffs
失真模型的参数向量

T_cn_cnm1
相机外在转换，总是相对于链中的最后一个相机
（例如cam1：T_cn_cnm1 = T_c1_c0，将cam0转换为cam1坐标）

T_cam_imu
IMU extrinsics：从IMU到相机坐标的转换（T_c_i）

timeshift_cam_imu
相机和IMU时间戳之间的时间间隔，以秒为单位（t_imu = t_cam + shift）

rostopic
摄像机图像流的主题

resolution
相机分辨率[width,height]
~~~

2. `imu_adis16448.yaml`

~~~sh
#Accelerometers
accelerometer_noise_density: 1.86e-03   #Noise density (continuous-time)
accelerometer_random_walk:   4.33e-04   #Bias random walk

#Gyroscopes
gyroscope_noise_density:     1.87e-04   #Noise density (continuous-time)
gyroscope_random_walk:       2.66e-05   #Bias random walk

rostopic:                    /imu0      #the IMU ROS topic
update_rate:                 200.0      #Hz (for discretization of the values above)
~~~


# 4 激光与相机联合标定

分为：基于目标，基于无目标，联合


## 4.1 `cam_lidar_calibration`功能包

[csdn参考链接](https://blog.csdn.net/weixin_41681988/article/details/122867113?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167093498116782427450816%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167093498116782427450816&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-122867113-null-null.142^v68^control,201^v4^add_ask,213^v2^t3_esquery_v3&utm_term=cam_lidar_calibration&spm=1018.2226.3001.4187)

1. 原理：
激光坐标系下得标定板法向量乘以外参 = 相机坐标系下标定板的法向量

相机与激光联合标定，需要标定板子，手动滤掉激光点，

2. 依赖

需要以下库：

~~~sh
opencv, pcl # 源码编译
ros-noetic-tf2-sensor-msgs # apt
pandas scipy pandas matplotlib rosbag # pip 、scipy也可以apt
~~~


+ `ubuntu18.04` 中 `opencv-3.2.0,ubuntu20.04` 中用 `opencv-3.4.16与opencv-4.6.0` 已成功运行该功能包

+ `ubuntu18.04` 需要 `python2和opencv3.2` , 本人已测试成功

+ `ubuntu18.04` 中 `opencv-3.4.16与opencv-4.6.0` 运行时,`opencv` 会报错

+ `agx` 板子上 `ubuntu18.04` 安装 `opencv-3.2.0` 带  `opencv-contrib` 会报错,可以只编译 `opencv-3.2.0`

+ 我单独编译过 `pcl-1.10.0` 库，这里就安装`ros-[ros版本noetic,melodic]-tf2-sensor-msgs`

---

~~~sh
# ubuntu20.04版本noetic
sudo apt update 
sudo apt-get install -y ros-noetic-pcl-conversions ros-noetic-pcl-ros ros-noetic-tf2-sensor-msgs
~~~

这里基本使用过ros中python代码，基本都满足，可以直接先编译，差什么依赖安装什么

~~~sh
# Ubuntu20.04默认为python3
sudo apt install python3-pip ros-noetic-tf2-sensor-msgs  
pip3 install pandas scipy
~~~

+ **`ubnntu18.04`** 环境 **`python2.7`** 与 **`opencv3.2.0`**

~~~sh
# 依赖
sudo apt install ros-melodic-tf2-sensor-msgs 

pip install pandas scipy pandas matplotlib rosbag
~~~

+ agx板子**`ubnntu18.04`**

前面加上python2.7环境
~~~sh
# scipy
sudo apt-get install python-scipy ros-melodic-tf2-sensor-msgs 
# pandas matplotlib rosbag
/usr/bin/python2.7 -m pip install pandas matplotlib rosbag
~~~

+ 运行前需要修改 **`scripts/visualise_results.py`** 脚本中python的路径

18.04 才需要, 20.04不需要用,默认是python3

~~~python
# 修改visualise_results.py 中的python环境
vim vim scripts/visualise_results.py 

# 默认的20.04
#!/usr/bin/env python

# 18.04conda虚拟环境
#!/home/lin/miniconda3/envs/python2.7/bin/python

# agv中python环境
#!/usr/bin/env python2.7
~~~

---

3. 下载编译
~~~sh
cd cam_lidar_calibration/src
git clone https://github.com/acfr/cam_lidar_calibration.git
cd .. 
~~~

编译
~~~sh
catkin build cam_lidar_calibration -DCMAKE_BUILD_TYPE=Release -j4
~~~

4. 测试

在源码中，作者该出了测试的标定数据集，输入命令进行测试标定：

~~~sh
roslaunch cam_lidar_calibration run_optimiser.launch import_samples:=true
~~~

该程序根据`cam_lidar_calibration_ws/src/cam_lidar_calibration/data/vlp/`文件夹下的`pose.csv`标定，在该文件夹生成一个标定`camera和lidar`外参文件，每一行则是迭代后的结果。

5. 获取评估校准结果

~~~sh
roslaunch cam_lidar_calibration assess_results.launch csv:=$(rospack find cam_lidar_calibration)/data/vlp/calibration_2023-01-04_20-34-21.csv visualise:true
~~~



### 4.1.1 标定自己的`camera和lidar`


1. 录制数据包或者在线启动传感器

这里选择在线启动传感器,标定板悬空放置

![](img/pose3.png)


标定自己的`camera和lidar`,主要修改`cam_lidar_calibration/cfg/camera_info.yaml`和`params.yaml`

2. `camera_info.yaml`:
设置是否为鱼眼相机、像素宽和高、内参矩阵和失真系数。相机参数自行标定

[ROS官方标定教程](http://wiki.ros.org/camera_calibration) 也可以参考第2节的相机标定方法

~~~sh
distortion_model: "non-fisheye"
 width: 640
height: 480
D: [0,0,0,0]
K: [617.68,0.0,325.963,0.0,617.875,242.513,0.0,0.0,1]
~~~

3. `params.yaml`

~~~sh
# Topics
camera_topic: "/camera/color/image_raw"
camera_info: "/camera/color/camera_info"
lidar_topic: "/velodyne_points"

#Dynamic rqt_reconfigure default bounds,点云的选取范围
feature_extraction:
x_min: -10.0
x_max: 10.0
y_min: -8.0
y_max: 8.0
z_min: -5.0
z_max: 5.0

# Properties of chessboard calibration target
chessboard:
pattern_size:  #棋盘的内部顶点7*5
height: 7
width: 5  
square_length: 95 #棋盘格的长度mm
board_dimension:  # 安装棋盘打印的背板的宽度和高度。
width: 594
height: 897 
translation_error: #棋盘中心与背板中心的偏移量（见下图）。
x: 0
y: 0
~~~

棋盘中心与背板中心的偏移量

![](img/221214-1038.png)

4. 正式开始标定

**开启程序采集表定数据，运行命令：**

~~~sh
roslaunch cam_lidar_calibration run_optimiser.launch import_samples:=false
~~~

会出现`RVIZ和rqt_reconfigure`窗口，在`RVIZ中panels->display`修改相机的话题和激光雷达点云对应的`frame_id`。

**采集数据：分割出标定板的点云**

调整`rqt_reconfigure/feature_extraction的xyz最大值最小值`以使得标定板的点云和周围环境**分开**，

使其**仅显示棋盘**。如果棋盘没有完全隔离，可能会影响棋盘的平面拟合，还会导致棋盘尺寸误差较大。下图是过滤点云前后效果：

![](img/221219-1542.png)

在过滤周围环境点云后，在`rviz中**点击Capture sample**`采集样本，会出线绿色框代表根据点云拟合出来的标定板平面
![](img/221219-1543.png)


最好采集10个样本以上，再点击`rviz中的optimise`进行标定,

在优化过程中将会在`cam_lidar_calibration/data`生成当前时间日期的文件夹，`存放采集的图像、点云pcd、位姿，标定后camer和lidar外参文件`.

---