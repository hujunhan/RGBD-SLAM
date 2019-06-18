//
// Created by hu on 2019/6/10.
//

#ifndef RGBD_SLAM_DATA_H
#define RGBD_SLAM_DATA_H


#include "common_include.h"
#include "chrono"
#include "sys/timeb.h"

using namespace std::chrono;
namespace myslam {
    namespace data {
        void read_info(string path, vector<string> &rgb_file, vector<string> &depth_file) {
            ifstream fin(path + "info.txt");

            if (!fin) {
                cout << "please generate the associate file called info.txt!" << endl;
                return;
            }
            while (!fin.eof()) {
                string file_name;
                fin >> file_name;
                rgb_file.push_back(path + "/rgb/" + file_name);
                depth_file.push_back(path + "/depth/" + file_name);
            }
            rgb_file.pop_back();
            depth_file.pop_back();
            fin.close();
        }

        void write_traj(string path, vector<SE3> tj) {
            ofstream fout(path + "traj.txt");
            for (size_t i = 0; i < tj.size(); i++) {
                auto m = tj[i].translation();
                fout << m.transpose() << endl;
            }
            fout.close();
        }

        unsigned long long getUTCtime(void) {
            unsigned long long microseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            struct timeb t1;
            ftime(&t1);
            auto second = t1.time;
            auto milltm = t1.millitm;

            return microseconds_since_epoch;

        }


    }
}
#endif //RGBD_SLAM_DATA_H
