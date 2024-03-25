#ifndef POINTXYZIR_H
#define POINTXYZIR_H
#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>

namespace pcl
{
struct PointXYZIR
{
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
}  // namespace pcl

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR, (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                                      intensity)(uint16_t, ring, ring))

#endif
