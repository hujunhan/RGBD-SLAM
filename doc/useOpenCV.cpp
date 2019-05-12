//
// Created by hu on 2019/5/12.
//
#include "librealsense2/rs.hpp"
#include "opencv2/highgui.hpp"
#include "iostream"
using namespace cv;
int main(void)
{
    rs2::colorizer color_map;
    rs2::pipeline p;
    p.start();
    rs2::frameset frames=p.wait_for_frames();
    for(int i = 0; i < 30; i++)
    {
        //Wait for all configured streams to produce a frame
        frames = p.wait_for_frames();
    }
    auto color=frames.get_color_frame();
    auto color_image=color.get_data();
    auto depth=frames.get_depth_frame().apply_filter(color_map);
    auto depth_image=color.get_data();
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();
    Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
    std::cout<< h;
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);

    return 0;
}
