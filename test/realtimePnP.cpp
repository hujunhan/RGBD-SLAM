#include "Windows.h"
#include "librealsense2/rs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/datasets/slam_tumindoor.hpp"
#include "iostream"
#include "common_include.h"
//#include "myslam/camera.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;

std::vector<std::string> read_directory(const std::string &path);

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

void pose_estimation_3d3d(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
        Mat &R, Mat &t
);

int main(void) {
//    myslam::Camera::Ptr camera(new myslam::Camera("../config.yaml"));
    int window_h=800;
    int window_w=800;
    YAML::Node config = YAML::LoadFile("../config.yaml");
    string dataset_path = config["test"]["dataset_path"].as<string>();
    rs2::pipeline p;
    rs2::config cfg;
    //使用BRG8才能显示正常的颜色，因为OpenCV就是这样规定显示的
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    auto profile = p.start(cfg);
    rs2::frameset frames;
    for (int i = 0; i < 10; i++) {
        //Wait for all configured streams to produce a frame
        frames = p.wait_for_frames();
    }

    auto camera_name = config["test"]["camera"].as<string>();
    auto camera_ins = config["camera"][camera_name]["instrinsic"];

    vector<SE3> tj;
    Eigen::Matrix<double, 3, 3> R;
    R << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    cout << "R: " << endl << R << endl;
    Eigen::Matrix<double, 3, 1> t;
    t << 0, 0, 0;
    cout << "t: " << endl << t << endl;
    Sophus::SE3<double> SE3_Rt(R, t);   // Create Sophus SE3 from R and t
    tj.push_back(SE3_Rt);

    vector<Mat> t_traj;
    t_traj.push_back((Mat_<double>(3, 1) << 0, 0, 0));
    //防止溢出
    frames = p.wait_for_frames();
    auto depth = frames.get_depth_frame();
    auto color = frames.get_color_frame();
    Mat img_1(Size(640, 480), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);
    Mat depth_1(Size(640, 480), CV_16UC1, (void *) depth.get_data(), Mat::AUTO_STEP);
    int i=0;
    SYSTEMTIME sys;
    Mat traj_mat(window_h,window_w,CV_8UC3,Scalar(0,0,0));
    int last_x=0.5*window_h;
    int last_y=0.5*window_w;
    namedWindow("Trajectory",WINDOW_AUTOSIZE);
    while(true) {

        GetLocalTime(&sys);
        stringstream s;
        s << sys.wSecond;
        s << sys.wMilliseconds;
        cout<<"start at"<<s.str()<<endl;
        frames = p.wait_for_frames();
        frames = align_to_color.process(frames);
        auto depth = frames.get_depth_frame();
        auto color = frames.get_color_frame();
        Mat img_2(Size(640, 480), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);
        Mat depth_2(Size(640, 480), CV_16UC1, (void *) depth.get_data(), Mat::AUTO_STEP);
        //load all the image we need

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
        int match_count = 0;
        double sumdd = 0;
        for (DMatch m:matches) {
            ushort d = depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(
                    keypoints_1[m.queryIdx].pt.x)];
            if (d == 0)   // bad depth
                continue;
            float dd = d / 1000.0;
            sumdd += dd;
            Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
            pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
            pts_2d.push_back(keypoints_2[m.trainIdx].pt);
            match_count++;
        }
        auto averagedd = sumdd / match_count;
//        cout<<i<<" average dd: "<<sumdd/match_count<<endl;


        Mat r, t;
        Mat R;
        solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t, false);
        cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵



        Eigen::Matrix3d R_e;
        R_e << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
        Sophus::SO3<double> SO3_R(R_e);
        auto x = t.at<double>(0, 0);
        auto y = t.at<double>(0, 1);
        auto z = t.at<double>(0, 2);
        auto length = sqrt(x * x + y * y + z * z);



        auto T = Sophus::SE3<double>(
                SO3_R,
                Vec3(x, y, z));
//        if (abs(T.angleY()) > config["constrain"]["angle"].as<float>())
//            continue;
        if (length > config["constrain"]["distance"].as<float>()) {
            continue;
        }

        cout << i++ << " length:" << length << " angle:" << T.angleY() << " 3dpair:" << match_count << endl;
        GetLocalTime(&sys);
        stringstream a;
        a << sys.wSecond;
        a << sys.wMilliseconds;
        cout<<"end at"<<a.str()<<endl;
        cout<<"--------------------"<<endl;
        tj.push_back(tj.back() * T);
        auto current_point=tj.back().translation();
        int cur_x=current_point[0]*100+0.5*window_h;
        int cur_y=current_point[2]*100+0.5*window_w;
        line(traj_mat,Point(last_x,last_y),Point(cur_x,cur_y),Scalar(0,0,255));
        last_x=cur_x;
        last_y=cur_y;
        img_2.copyTo(img_1);
        depth_2.copyTo(depth_1);
        imshow("Trajectory",traj_mat);
        if(waitKey(1)=='q')
            break;
    }

    fstream fout(dataset_path + "traj.txt", ios::app);
    for (size_t i = 0; i < tj.size(); i++) {
        auto m = tj[i].translation();
        fout << m.transpose() << endl;
    }

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
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
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
//    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
    Point2d principal_point(318.138, 241.882);  //相机光心, TUM dataset标定值
    double focal_length = 615;      //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
//    cout << "essential_matrix is " << endl << essential_matrix << endl;

    //-- 计算单应矩阵
    //-- 但是本例中场景不是平面，单应矩阵意义不大
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
//    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    // 此函数仅在Opencv3中提供
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
//    cout << "R is " << endl << R << endl;
//    cout << "t is " << endl << t << endl;

}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}


