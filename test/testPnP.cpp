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
#include "myslam/camera.h"

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
    myslam::Camera::Ptr camera(new myslam::Camera("../config.yaml"));

    YAML::Node config = YAML::LoadFile("../config.yaml");
    string dataset_path = config["test"]["dataset_path"].as<string>();

    auto camera_name = config["test"]["camera"].as<string>();
    auto camera_ins = config["camera"][camera_name]["instrinsic"];
    int data_size = config["test"]["data_size"].as<int>();
    vector<string> rgb_file, depth_file;
    ifstream fin(dataset_path + "info.txt");
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
    vector<SE3<double>> tj;
    Eigen::Matrix<double, 3, 3> R;
    R  << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    cout << "R: " << endl << R << endl;
    Eigen::Matrix<double, 3, 1> t;
    t  << 0, 0, 0;
    cout << "t: " << endl << t << endl;
    Sophus::SE3<double> SE3_Rt(R, t);   // Create Sophus SE3 from R and t
    tj.push_back(SE3_Rt);
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
//    vector<Mat> t_traj;
    t_traj.push_back((Mat_<double>(3, 1) << 0, 0, 0));
    //防止溢出
    if (data_size > depth_file.size())
        data_size = depth_file.size();
    for (size_t i = 160; i < (data_size - 1); i++) {
//       cout << "The " << i + 1 << " image path is " << depth_file[i] << endl;

        //load all the image we need
        Mat img_1 = imread(rgb_file[i]);
        Mat img_2 = imread(rgb_file[i + 1]);
        Mat depth_1 = imread(depth_file[i], IMREAD_ANYDEPTH);
        Mat depth_2 = imread(depth_file[i + 1], IMREAD_ANYDEPTH);

        //ORB
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches;
        find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
//        cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

        //相机内参
        Mat K = (Mat_<double>(3, 3)
                << camera_ins["fx"].as<double>(), 0, camera_ins["cx"].as<double>(), 0, camera_ins["fy"].as<double>(), camera_ins["cy"].as<double>(), 0, 0, 1);
        vector<Point3f> pts_3d;
        vector<Point2f> pts_2d;
        for (DMatch m:matches) {
            ushort d = depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(
                    keypoints_1[m.queryIdx].pt.x)];
            if (d == 0)   // bad depth
                continue;
            float dd = d / 1000.0;
            Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
            pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
            pts_2d.push_back(keypoints_2[m.trainIdx].pt);
        }


        Mat r, t;
        solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
        Mat R;
        cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
//        Eigen::Matrix3d R_e;
//        R_e << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
//                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
//                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
//        Sophus::SO3<double> SO3_R(R_e);
//        auto T=Sophus::SE3<double>(
//                SO3_R,
//                Vector3d(t.at<double>(0,0),t.at<double>(0,1),t.at<double>(0,2)));
//        tj.push_back(tj.back()*T);
        t_traj.push_back(t + t_traj.back());
    }
//    GetLocalTime(&sys);
//    stringstream a;
//    a << sys.wSecond;
//    a << sys.wMilliseconds;
//    cout << s.str() << endl;
//    cout << a.str() << endl;
//    cout << "done";
    fstream fout(dataset_path + "traj.txt", ios::app);
    for (size_t i = 0; i < t_traj.size(); i++) {
        auto m = t_traj[i];
        double x = m.at<double>(0, 0);
        double y = m.at<double>(0, 1);
//        double z = m.at<double>(0, 2);

//        if((x*x+y*y+z*z)>0.1)
//            continue;
        fout << x<< "\t";
        fout <<y << "\t";
//        fout << 0 << "\t";

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
        if (match[i].distance <= max(3 * min_dist, 30.0)) {
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
    R_(0, 2) = 0;
    R_(1, 2) = 0;
    R_(2, 2) = 1;
    R_(2, 1) = 0;
    R_(2, 0) = 0;
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
//    R = (Mat_<double>(3, 3) <<
//            R_(0, 0), R_(0, 1), R_(0, 2),
//            R_(1, 0), R_(1, 1), R_(1, 2),
//            R_(2, 0), R_(2, 1), R_(2, 2)
//    );
    R = (Mat_<double>(3, 3) <<
                            R_(0, 0), R_(0, 1), 0,
            R_(1, 0), R_(1, 1), 0,
            0, 0, 1
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}


