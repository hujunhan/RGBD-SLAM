#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"
namespace myslam {
    namespace camera {
        Point2d pixel2cam(const Point2d &p, const Mat &K) {
            return Point2d
                    (
                            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
                    );
        }

       SE3 get_init_T(void)
       {
           Eigen::Matrix<double, 3, 3> R;
           R << 1, 0, 0, 0, 1, 0, 0, 0, 1;
           cout << "R: " << endl << R << endl;
           Eigen::Matrix<double, 3, 1> t;
           t << 0, 0, 0;
           cout << "t: " << endl << t << endl;
           Sophus::SE3<double> SE3_Rt(R, t);   // Create Sophus SE3 from R and t
           return SE3_Rt;
       }
       SE3 Rt2T(Mat R,Mat t)
       {
           Eigen::Matrix3d R_e;
           R_e << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                   R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                   R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
           Sophus::SO3<double> SO3_R(R_e);
           auto x = t.at<double>(0, 0);
           auto y = t.at<double>(0, 1);
           auto z = t.at<double>(0, 2);
           auto T = SE3(
                   SO3_R,
                   Vec3(x, y, z));
           return T;
       }
       double calc_t_length(Mat t)
       {
           auto x = t.at<double>(0, 0);
           auto y = t.at<double>(0, 1);
           auto z = t.at<double>(0, 2);
           auto length = sqrt(x * x + y * y + z * z);
           return length;
       }

    }
}  // namespace myslam
#endif  // MYSLAM_CAMERA_H
