
#include "librealsense2/rs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/datasets/slam_tumindoor.hpp"
#include "iostream"
#include "common_include.h"
#include "myslam/visual_odometry.h"
#include "myslam/camera.h"
#include "myslam/data.h"
#include "iomanip"
//#include "myslam/camera.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;


int main(void) {
//    myslam::Camera::Ptr camera(new myslam::Camera("../config.yaml"));
    int window_h = 800;
    int window_w = 800;
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
    for (int i = 0; i < 30; i++) {
        //Wait for all configured streams to produce a frame
        frames = p.wait_for_frames();
    }

    auto camera_name = config["test"]["camera"].as<string>();
    auto camera_ins = config["camera"][camera_name]["instrinsic"];

    vector<SE3> tj;
    auto init_pose = myslam::camera::get_init_T();
    tj.push_back(init_pose);
    vector<Mat> r_save;
    vector<Mat> t_save;
    //防止溢出
    frames = p.wait_for_frames();
    auto depth = frames.get_depth_frame();
    auto color = frames.get_color_frame();
    Mat img_1(Size(640, 480), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);
    Mat depth_1(Size(640, 480), CV_16UC1, (void *) depth.get_data(), Mat::AUTO_STEP);
    int i = 0;

    Mat traj_mat(window_h, window_w, CV_8UC3, Scalar(0, 0, 0));
    int last_x = 0.5 * window_h;
    int last_y = 0.5 * window_w;
    namedWindow("Trajectory", WINDOW_AUTOSIZE);
    Mat r, t;
    Mat R;
    vector<unsigned long long> utc_time;
    FileStorage rfs(dataset_path + "r.yml", FileStorage::WRITE);
    FileStorage tfs(dataset_path + "t.yml", FileStorage::WRITE);
    while (true) {
        utc_time.push_back(myslam::data::getUTCtime());
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
        myslam::vo::find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
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
            Point2d p1 = myslam::camera::pixel2cam(keypoints_1[m.queryIdx].pt, K);
            pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
            pts_2d.push_back(keypoints_2[m.trainIdx].pt);
            match_count++;
        }
        auto averagedd = sumdd / match_count;
//        cout<<i<<" average dd: "<<sumdd/match_count<<endl;



        solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t, true);
        rfs << "r" + to_string(i) << r;
        tfs << "t" + to_string(i) << t;
        cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵

        auto T = myslam::camera::Rt2T(R, t);
//        auto T = myslam::camera::normalizeT(R, t);
        auto length = myslam::camera::calc_t_length(t);



//        if (abs(T.angleY()) > config["constrain"]["angle"].as<float>())
//            continue;
        if (length > config["constrain"]["distance"].as<float>()) {
            continue;
        }

        cout << i++ << " length:" << length << " angle:" << T.angleY() << " 3dpair:" << match_count << endl;
        tj.push_back(tj.back() * T);
        auto current_point = tj.back().translation();
        int cur_x = current_point[0] * 100 + 0.5 * window_h;
        int cur_y = current_point[2] * 100 + 0.5 * window_w;
        line(traj_mat, Point(last_x, last_y), Point(cur_x, cur_y), Scalar(0, 0, 255));
        last_x = cur_x;
        last_y = cur_y;
        img_2.copyTo(img_1);
        depth_2.copyTo(depth_1);
        imshow("Trajectory", traj_mat);
        if (waitKey(1) == 'q')
            break;

    }

    tfs.release();
    rfs.release();

    ofstream fout(dataset_path + "traj.txt");
    for (size_t i = 0; i < tj.size(); i++) {
        auto m = tj[i].translation();
        fout << setiosflags(ios::fixed) << setprecision(3) << (utc_time[i] * 1.0 / 1000) << " ";
        fout << setiosflags(ios::fixed) << setprecision(6) << m.transpose()[0] << " " << m.transpose()[2] << endl;
    }
    fout.close();

}





