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
#include "fstream"
#include "common_include.h"
using namespace cv;
using namespace cv::datasets;
std::vector <std::string> read_directory( const std::string& path  );
int main(void)
{
    string dataset_path="../data/dormitory";

    vector<string> rgb_file,depth_file;
    ifstream fin (dataset_path+"/info.txt");
    if ( !fin )
    {
        cout<<"please generate the associate file called info.txt!"<<endl;
        return 1;
    }
    while (!fin.eof())
    {
        string file_name;
        fin>>file_name;
        cout<<"Img name: "<<file_name<<endl;
        rgb_file.push_back(dataset_path+"/rgb/"+file_name);
        depth_file.push_back(dataset_path+"/depth"+file_name);
    }
    rgb_file.pop_back();
    depth_file.pop_back();
    cout<<rgb_file.size();

    return 0;
}

