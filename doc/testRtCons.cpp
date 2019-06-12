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


void pose_estimation_3d3d(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
        Mat &R, Mat &t
);

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
    vector<double> y;
    y.push_back(_Pi/2);
    double len_sum=0;
    double angle_sum=0;
    vector<SE3> tj;
    vector<float> length_save;
    vector<float> angle_save;
    auto init_pose = myslam::camera::get_init_T();
    tj.push_back(init_pose);
    //防止溢出
    if (data_size > depth_file.size())
        data_size = depth_file.size();
    FileStorage rfs(dataset_path + "r.yml", FileStorage::READ);
    FileStorage tfs(dataset_path + "t.yml", FileStorage::READ);
    for (size_t i = 0; i < (data_size - 1); i++) {
        Mat r,t;
        Mat R;
        rfs["r"+to_string(i)]>>r;
        tfs["t"+to_string(i)]>>t;
        cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵

//        auto T = myslam::camera::Rt2T(R, t);
        auto T = myslam::camera::normalizeTwithLength(R, t,length_save);
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
        if(i<300)
        {
            len_sum+=length;
            angle_sum+=T.angleY();
        }
        length_save.push_back(length);
        angle_save.push_back(T.angleY());
        y.push_back(y.back()+T.angleY());
        if (i % 100 == 0) {
            cout << i << " length:" << length << " angle:" << T.angleY() << " 3dpair:" << endl;
        }
//        cout<<myslam::data::getUTCtime()<<endl;
        tj.push_back(tj.back() * T);
    }
    cout<<"len sum"<<len_sum/300<<endl;
    cout<<"angle sum"<<angle_sum/300<<endl;
    myslam::data::write_traj(dataset_path, tj);
    ofstream fout(dataset_path + "coor&y.txt");
    for (size_t i = 0; i < length_save.size(); i++) {
        auto m=tj[i].translation();
        fout <<m.transpose() << " " << y[i] << endl;
    }
    fout.close();
    ofstream foutl(dataset_path + "length&angle.txt");


    for (size_t i = 0; i < length_save.size(); i++) {
        fout <<length_save[i] << " " <<angle_save[i] << endl;
    }
    fout.close();
    return 0;
}