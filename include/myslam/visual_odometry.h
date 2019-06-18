//
// Created by hu on 2019/6/10.
//

#ifndef RGBD_SLAM_VISUAL_ODOMETRY_H
#define RGBD_SLAM_VISUAL_ODOMETRY_H

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
namespace myslam {
    namespace vo {
        bool compareD(const DMatch &a, const DMatch &b) {
            return a.distance < b.distance;
        }

        void find_feature_matches(const Mat &img_1, const Mat &img_2,
                                  std::vector<KeyPoint> &keypoints_1,
                                  std::vector<KeyPoint> &keypoints_2,
                                  std::vector<DMatch> &matches) {
            //-- 初始化
            Mat descriptors_1, descriptors_2;
            // used in OpenCV3

            Ptr<FeatureDetector> detector = ORB::create();
            Ptr<DescriptorExtractor> descriptor = ORB::create();

            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
//    cv::FlannBasedMatcher matcher;
            //-- 第一步:检测 Oriented FAST 角点位置

            detector->detect(img_1, keypoints_1);
            detector->detect(img_2, keypoints_2);

            //-- 第二步:根据角点位置计算 BRIEF 描述子
            descriptor->compute(img_1, keypoints_1, descriptors_1);
//    descriptors_1.convertTo(descriptors_1,CV_32F);
            descriptor->compute(img_2, keypoints_2, descriptors_2);
//    descriptors_2.convertTo(descriptors_2,CV_32F);
            //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
            vector <DMatch> match;
            // BFMatcher matcher ( NORM_HAMMING );
            matcher->match(descriptors_1, descriptors_2, match);
            sort(match.begin(),match.end(),compareD);
            for (int i=0;i<100 ;i++)
            {
                matches.push_back(match[i]);
            }
//            cout<<matches.size()<<endl;
            //-- 第四步:匹配点对筛选
//            double min_dist = 10000, max_dist = 0;
//
////            找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
//            for (int i = 0; i < descriptors_1.rows; i++) {
//                double dist = match[i].distance;
//                if (dist < min_dist) min_dist = dist;
//                if (dist > max_dist) max_dist = dist;
//            }
//
////    printf("-- Max dist : %f \n", max_dist);
////    printf("-- Min dist : %f \n", min_dist);
////            当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
//            for (int i = 0; i < descriptors_1.rows; i++) {
//                if (match[i].distance <= max(2 * min_dist, 30.0)) {
//                    matches.push_back(match[i]);
//                }
//            }
//            Mat img_goodmatch;
//            drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_goodmatch);
//            imshow("matchers", img_goodmatch);
//            if(waitKey(0)=='q')
//                return;
        }
        void OptFlow(const cv::Mat &ref_img,
                     const cv::Mat &curImg,
                     const std::vector<Point2f> featurePoints,
                     const std::vector<Point3f> &landmarks,
                     std::vector<Point2f> &featurePoints_ref,
                     std::vector<Point3f> &landmarks_ref) {
            vector<Point2f> nextPts;
            vector<uchar> status;
            vector<float> err;

            calcOpticalFlowPyrLK(ref_img, curImg, featurePoints, nextPts, status, err);

            for (int  j = 0; j < status.size(); j++) {
                if (status[j] == 1) {
                    featurePoints_ref.push_back(nextPts[j]);
                    landmarks_ref.push_back(landmarks[j]);
                }
            }

        }
        void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                                  std::vector<KeyPoint> keypoints_2,
                                  std::vector<DMatch> matches,
                                  Mat &R, Mat &t) {
            // 相机内参,TUM Freiburg2

            //-- 把匹配点转换为vector<Point2f>的形式
            vector <Point2f> points1;
            vector <Point2f> points2;

            for (int i = 0; i < (int) matches.size(); i++) {
                points1.push_back(keypoints_1[matches[i].queryIdx].pt);
                points2.push_back(keypoints_2[matches[i].trainIdx].pt);
            }

            //-- 计算基础矩阵
            Mat fundamental_matrix;
            fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);

            //-- 计算本质矩阵
            Point2d principal_point(318.138, 241.882);  //相机光心, TUM dataset标定值
            double focal_length = 615;      //相机焦距, TUM dataset标定值
            Mat essential_matrix;
            essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

            //-- 计算单应矩阵
            Mat homography_matrix;
            homography_matrix = findHomography(points1, points2, RANSAC, 3);

            //-- 从本质矩阵中恢复旋转和平移信息.
            recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);


        }

        void pose_estimation_3d3d(
                const vector <Point3f> &pts1,
                const vector <Point3f> &pts2,
                Mat &R, Mat &t
        ) {
            Point3f p1, p2;     // center of mass
            int N = pts1.size();
            for (int i = 0; i < N; i++) {
                p1 += pts1[i];
                p2 += pts2[i];
            }
            p1 = Point3f(Vec3f(p1) / N);
            p2 = Point3f(Vec3f(p2) / N);
            vector <Point3f> q1(N), q2(N); // remove the center
            for (int i = 0; i < N; i++) {
                q1[i] = pts1[i] - p1;
                q2[i] = pts2[i] - p2;
            }

            // compute q1*q2^T
            Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
            for (int i = 0; i < N; i++) {
                W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) *
                     Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
            }
//            cout<<"W="<<W<<endl;

            // SVD on W
            Eigen::JacobiSVD <Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            if (U.determinant() * V.determinant() < 0) {
                for (int x = 0; x < 3; ++x) {
                    U(x, 2) *= -1;
                }
            }

//            cout<<"U="<<U<<endl;
//            cout<<"V="<<V<<endl;

            Eigen::Matrix3d R_ = U * (V.transpose());
            Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

            // convert to cv::Mat
            R = (Mat_<double>(3, 3) <<
                                    R_(0, 0), R_(0, 1), R_(0, 2),
                    R_(1, 0), R_(1, 1), R_(1, 2),
                    R_(2, 0), R_(2, 1), R_(2, 2)
            );
            t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
        }


    }
}
#endif //RGBD_SLAM_VISUAL_ODOMETRY_H
