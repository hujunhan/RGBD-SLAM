//
// Created by hu on 2019/5/15.
//
//
// Created by hu on 2019/5/12.
//
#include "Windows.h"
#include "librealsense2/rs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/datasets/slam_tumindoor.hpp"
#include "iostream"
#include "fstream"

#include "common_include.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;

std::vector<std::string> read_directory(const std::string &path);

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

void pose_estimation_3d3d(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
        Mat &R, Mat &t
);

int main(void) {

    string dataset_path = "../data/dormitory";

    vector<string> rgb_file, depth_file;
    ifstream fin(dataset_path + "/info.txt");
    if (!fin) {
        cout << "please generate the associate file called info.txt!" << endl;
        return 1;
    }
    while (!fin.eof()) {
        string file_name;
        fin >> file_name;
//        cout<<"Img name: "<<file_name<<endl;
        rgb_file.push_back(dataset_path + "/rgb/" + file_name);
        depth_file.push_back(dataset_path + "/depth/" + file_name);
    }
    rgb_file.pop_back();
    depth_file.pop_back();
    cout << "The size of the dataset is " << rgb_file.size() << endl;
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    stringstream s;
    s << sys.wSecond;
    s << sys.wMilliseconds;

    vector<Mat> t_traj;
    t_traj.push_back((Mat_<double>(3, 1) << 0, 0, 0));

    for (size_t i = 0; i < (rgb_file.size() - 1); i++) {
//       cout << "The " << i + 1 << " image path is " << depth_file[i] << endl;

        //load all the image we need
        Mat img_1 = imread(rgb_file[i]);
        Mat img_2 = imread(rgb_file[i + 1]);
        Mat depth_1 = imread(depth_file[i],IMREAD_ANYDEPTH);
        Mat depth_2 = imread(depth_file[i + 1],IMREAD_ANYDEPTH);

        //ORB
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches;
        find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
//        cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

        //内参
        //freiburg2
//        Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
        Mat K = (Mat_<double>(3, 3) << 615, 0, 318, 0, 615, 318, 0, 0, 1);
        vector<Point3f> pts1, pts2;
        for (DMatch m:matches) {
            ushort d1 = depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(
                    keypoints_1[m.queryIdx].pt.x)];
            ushort d2 = depth_2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(
                    keypoints_2[m.trainIdx].pt.x)];
            if (d1 == 0 || d2 == 0)   // bad depth
                continue;
            Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
            Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
            float dd1 = float(d1) / 1000.0;
            float dd2 = float(d2) / 1000.0;
            pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
            pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
//            cout<<"3d-3d pairs: "<<pts1.size() <<endl;

        }
        Mat R, t;
        pose_estimation_3d3d(pts1, pts2, R, t);
//            cout<<"ICP via SVD results: "<<endl;
//            cout<<"R = "<<R<<endl;
        cout << "t = " << t << endl;
        cout << "traj = "<<t_traj.back() << endl;
        t_traj.push_back(t+t_traj.back());
    }
//    GetLocalTime(&sys);
//    stringstream a;
//    a << sys.wSecond;
//    a << sys.wMilliseconds;
//    cout << s.str() << endl;
//    cout << a.str() << endl;
//    cout << "done";
    fstream fout("../data/dormitory/traj.txt", ios::app);
    for (size_t i = 0; i < t_traj.size(); i++) {
        auto m = t_traj[i];
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                fout << m.at<double>(i, j) << "\t";
            }

        }
        fout << std::endl;

    }
    fout.close();
    return 0;
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
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void pose_estimation_3d3d(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
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
    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
//    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (U.determinant() * V.determinant() < 0) {
        for (int x = 0; x < 3; ++x) {
            U(x, 2) *= -1;
        }
    }

//    cout << "U=" << U << endl;
//    cout << "V=" << V << endl;

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


