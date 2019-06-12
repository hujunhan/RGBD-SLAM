#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"

namespace myslam {
    namespace camera {
        float average_vector(vector<float> somev, int num) {
            float sum = 0;
            int len=somev.size();
            for (int i = 0; i < num; i++) {
                sum += somev[len-i-1];
            }
            return sum / num;
        }
        Point2d pixel2cam(const Point2d &p, const Mat &K) {
            return Point2d
                    (
                            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
                    );
        }

        SE3 get_init_T(void) {
            Eigen::Matrix<double, 3, 3> R;
            R << 1, 0, 0, 0, 1, 0, 0, 0, 1;
            cout << "R: " << endl << R << endl;
            Eigen::Matrix<double, 3, 1> t;
            t << 0, 0, 0;
            cout << "t: " << endl << t << endl;
            Sophus::SE3<double> SE3_Rt(R, t);   // Create Sophus SE3 from R and t
            return SE3_Rt;
        }

        SE3 Rt2T(Mat R, Mat t) {
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

        double calc_t_length(Mat t) {
            auto x = t.at<double>(0, 0);
            auto y = t.at<double>(0, 1);
            auto z = t.at<double>(0, 2);
            auto length = sqrt(x * x + y * y + z * z);
            return length;
        }

        SE3 normalizeT(Mat &R, Mat &t) {
            auto T = Rt2T(R, t);
            auto angle = T.angleY();
            auto length = calc_t_length(t);

            if (abs(angle) < 0.005) {
//                angle = angle / 2;
                angle = 0;
                auto x = t.at<double>(0, 0);
                auto y = t.at<double>(0, 1);
                auto z = t.at<double>(0, 2);
                auto eigent = Vec3(x, y, z);
                if (length > 0.01) {
                    eigent.normalize();
                    eigent = eigent * 0.01;
                }
                t.at<double>(0, 0) = eigent[0];
                t.at<double>(0, 1) = eigent[1];
                t.at<double>(0, 2) = eigent[2];

                auto R = Eigen::AngleAxisd(angle, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();
                Sophus::SE3<double> SE3_Rt(R, eigent);
                return SE3_Rt;
            } else {
                if (abs(angle) < 0.02)
                    angle = angle * 0.5;
                auto R = Eigen::AngleAxisd(angle, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();
                t.at<double>(0, 0) = 0;
                t.at<double>(0, 1) = 0;
                t.at<double>(0, 2) = 0;
                auto eigent = Vec3(0, 0, 0);
                Sophus::SE3<double> SE3_Rt(R, eigent);
                return SE3_Rt;
            }
        }
        SE3 normalizeTwithLength(Mat &R, Mat &t, vector<float> len_save) {
            auto T = Rt2T(R, t);
            auto angle = T.angleY();
            auto length = calc_t_length(t);

            if (abs(angle) < 0.005) {
//                angle = angle / 2;
                angle = 0;
                auto x = t.at<double>(0, 0);
                auto y = t.at<double>(0, 1);
                auto z = t.at<double>(0, 2);
                auto eigent = Vec3(x, y, z);
                if (length > 0.02) {
                    eigent.normalize();
                    eigent = eigent * average_vector(len_save,10);
                }
                t.at<double>(0, 0) = eigent[0];
                t.at<double>(0, 1) = eigent[1];
                t.at<double>(0, 2) = eigent[2];

                auto R = Eigen::AngleAxisd(angle, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();
                Sophus::SE3<double> SE3_Rt(R, eigent);
                return SE3_Rt;
            } else {
                if (abs(angle) < 0.02)
                    angle = angle * 0.5;
                auto R = Eigen::AngleAxisd(angle, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();
                t.at<double>(0, 0) = 0;
                t.at<double>(0, 1) = 0;
                t.at<double>(0, 2) = 0;
                auto eigent = Vec3(0, 0, 0);
                Sophus::SE3<double> SE3_Rt(R, eigent);
                return SE3_Rt;
            }
        }



//        SE3 correctT(Mat &R, Mat &t, vector<float> length_save, vector<float> angle_save) {
//            auto length_aver=average_vector(length_save,10);
//            auto angle_aver=average_vector(angle_save,10);
//        }



    }
}  // namespace myslam
#endif  // MYSLAM_CAMERA_H
