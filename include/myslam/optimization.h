//
// Created by hu on 2019/6/13.
//

#ifndef RGBD_SLAM_OPTIMIZATION_H
#define RGBD_SLAM_OPTIMIZATION_H

#include "common_include.h"
#include <ceres/ceres.h>
#include "ceres/rotation.h"
namespace myslam{
    namespace optimization{
        struct cost_function_define {
            cost_function_define(Point3d p1, Point3d p2) : _p1(p1), _p2(p2) {}
            template<typename T>
            bool operator()(const T *const cere_r, const T *const cere_t, T *residual) const {
                T p_1[3];
                T p_2[3];
                p_1[0] = T(_p1.x);
                p_1[1] = T(_p1.y);
                p_1[2] = T(_p1.z);
                ceres::AngleAxisRotatePoint(cere_r, p_1, p_2);
                p_2[0] = p_2[0] + cere_t[0];
                p_2[1] = p_2[1] + cere_t[1];
                p_2[2] = p_2[2] + cere_t[2];
                const T x = p_2[0] / p_2[2];
                const T y = p_2[1] / p_2[2];
                const T u = x * 615.3 + 318.1;
                const T v = y * 615.3 + 241.8;
                T p_3[3];
                p_3[0] = T(_p2.x);
                p_3[1] = T(_p2.y);
                p_3[2] = T(_p2.z);
                const T x1 = p_3[0] / p_3[2];
                const T y1 = p_3[1] / p_3[2];
                const T u1 = x1 * 615.3 + 318.1;
                const T v1 = y1 * 615.3 + 241.8;
                residual[0] = u - u1;
                residual[1] = v - v1;
                return true;
            }
            const Point3f _p1,_p2;
        };
    }
}
#endif //RGBD_SLAM_OPTIMIZATION_H
