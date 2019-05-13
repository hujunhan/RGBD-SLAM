//
// Created by hu on 2019/5/12.
//
#include "librealsense2/rs.hpp"
#include "opencv2/opencv.hpp"
#include "iostream"
using namespace cv;
int main(void) try
{
    rs2::colorizer color_map;
    rs2::pipeline p;
    rs2::config cfg;

    p.start();
    rs2::frameset frames;
    for(int i = 0; i < 30; i++)
    {
        //Wait for all configured streams to produce a frame
        frames = p.wait_for_frames();
    }

    auto color=frames.get_color_frame();
    auto depth=frames.get_depth_frame().apply_filter(color_map);

    const int w = color.as<rs2::video_frame>().get_width();
    const int h = color.as<rs2::video_frame>().get_height();
    Mat image(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    std::cout<<w<<std::endl;
    std::cout<<"width: "<<image.cols<<" height:"<<image.rows<<" channel: "<<image.channels()<<std::endl;
    waitKey(0);

    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
