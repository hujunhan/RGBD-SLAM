//
// Created by hu on 2019/5/15.
//
//
// Created by hu on 2019/5/12.
//
#include "librealsense2/rs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/datasets/slam_tumindoor.hpp"
#include "iostream"
#include "common_include.h"
//#include "myslam/camera.h"
#include "myslam/visual_odometry.h"
#include "myslam/camera.h"
#include "myslam/data.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;

std::vector<std::string> read_directory(const std::string &path);




int main(void) {


    cout << "VO start working! " << endl;
    YAML::Node config = YAML::LoadFile("../config.yaml");
    string dataset_path = config["test"]["dataset_path"].as<string>();
    auto camera_name = config["test"]["camera"].as<string>();
    auto camera_ins = config["camera"][camera_name]["instrinsic"];
    int data_size = config["test"]["data_size"].as<int>();

    vector<string> rgb_file, depth_file;
    myslam::data::read_info(dataset_path, rgb_file, depth_file);
    cout << "The path of the dataset is " << rgb_file[0] << endl;
    cout << "The size of the dataset is " << rgb_file.size() << endl;
    vector<float> dt;
    int len=rgb_file[0].length();
    cout<<"len "<<len<<endl;
    for (int i = 0; i < (rgb_file.size() - 2); i++) {
        float ddt=stof(rgb_file[i + 1].substr(len-12, 8)) - stof(rgb_file[i].substr(len-12, 8));
        dt.push_back(ddt);
//        cout<<ddt<<"\t";
    }

    vector<SE3> tj;
    vector<float> length_save;
    vector<float> angle_save;
    auto init_pose = myslam::camera::get_init_T();
    tj.push_back(init_pose);
    //防止溢出
    if (data_size > depth_file.size())
        data_size = depth_file.size();

    for (size_t i = 0; i < (data_size - 1); i++) {

        //load all the image we need
        Mat img_1 = imread(rgb_file[i]);
        Mat img_2 = imread(rgb_file[i + 1]);
        Mat depth_1 = imread(depth_file[i], IMREAD_ANYDEPTH);
        Mat depth_2 = imread(depth_file[i + 1], IMREAD_ANYDEPTH);

        //ORB
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches;
        myslam::vo::find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
        //相机内参
        Mat K = (Mat_<double>(3, 3)
                << camera_ins["fx"].as<double>(), 0, camera_ins["cx"].as<double>(), 0, camera_ins["fy"].as<double>(), camera_ins["cy"].as<double>(), 0, 0, 1);
        vector<Point3f> pts_3d;
        vector<Point2f> pts_2d;
        int match_count = 0;
        double sumdd = 0;
        float mindd = 1000;
        for (DMatch m:matches) {
            ushort d = depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(
                    keypoints_1[m.queryIdx].pt.x)];
            if (d == 0)   // bad depth
                continue;
            float dd = d / 1000.0;
            if (dd < mindd)
                mindd = dd;
            sumdd += dd;

            Point2d p1 = myslam::camera::pixel2cam(keypoints_1[m.queryIdx].pt, K);
            pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
            pts_2d.push_back(keypoints_2[m.trainIdx].pt);
            match_count++;
        }
        auto averagedd = sumdd / match_count;
//        cout<<i<<" average dd: "<<sumdd/match_count<<endl;


        Mat r, t;
        Mat R;
        solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t, false);
//        solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false, SOLVEPNP_EPNP);
        cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵

//        auto T = myslam::camera::Rt2T(R, t);
        auto T = myslam::camera::normalizeT(R, t);
        auto length = myslam::camera::calc_t_length(t);
//        cout<<"length "<<length<<endl;
//        if (abs(T.angleY()) > config["constrain"]["angle"].as<float>())
//            continue;

        if (length > config["constrain"]["distance"].as<float>()) {
            continue;
        }
        if (abs(T.angleY()) > config["constrain"]["angle"].as<float>()) {
            continue;
        }
        length_save.push_back(length);
        angle_save.push_back(T.angleY());
        if (i % 100 == 0) {
            cout << i << " length:" << length << " angle:" << T.angleY() << " 3dpair:" << match_count << " mindd:"
                 << mindd
                 << endl;
        }
//        cout<<myslam::data::getUTCtime()<<endl;
        tj.push_back(tj.back() * T);
    }

    myslam::data::write_traj(dataset_path, tj);
    ofstream fout(dataset_path + "length&angle1.txt");
    for (size_t i = 0; i < length_save.size(); i++) {
        fout << length_save[i] << " " << angle_save[i] << endl;
    }
    fout.close();
    return 0;
}




