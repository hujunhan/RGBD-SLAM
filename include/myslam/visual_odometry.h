//
// Created by hu on 2019/6/10.
//

#ifndef RGBD_SLAM_VISUAL_ODOMETRY_H
#define RGBD_SLAM_VISUAL_ODOMETRY_H

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
using namespace cv;
namespace myslam{
    namespace vo{
        bool compareD(const DMatch &a,const DMatch &b)
        {
            return a.distance<b.distance;
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
            vector<DMatch> match;
            // BFMatcher matcher ( NORM_HAMMING );
            matcher->match(descriptors_1, descriptors_2, match);
            sort(match.begin(),match.end(),compareD);
            for (int i=0;i<70;i++)
            {
                matches.push_back(match[i]);
            }
            //-- 第四步:匹配点对筛选
//    double min_dist = 10000, max_dist = 0;

            //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
//    for (int i = 0; i < descriptors_1.rows; i++) {
//        double dist = match[i].distance;
//        if (dist < min_dist) min_dist = dist;
//        if (dist > max_dist) max_dist = dist;
//    }

//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);
            //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
//    for (int i = 0; i < descriptors_1.rows; i++) {
//        if (match[i].distance <= max(2 * min_dist, 30.0)) {
//            matches.push_back(match[i]);
//        }
//    }
        }
        void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                                  std::vector<KeyPoint> keypoints_2,
                                  std::vector<DMatch> matches,
                                  Mat &R, Mat &t) {
            // 相机内参,TUM Freiburg2

            //-- 把匹配点转换为vector<Point2f>的形式
            vector<Point2f> points1;
            vector<Point2f> points2;

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


    }
}
#endif //RGBD_SLAM_VISUAL_ODOMETRY_H
