//
// Created by hu on 2019/5/12.
//
#include "librealsense2/rs.hpp"
#include "opencv2/opencv.hpp"
#include "iostream"

using namespace cv;
using namespace std;

int main(void) try {
    rs2::colorizer color_map;
    rs2::pipeline p;
    rs2::config cfg;
    //使用BRG8才能显示正常的颜色，因为OpenCV就是这样规定显示的
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    auto profile = p.start(cfg);

    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    double _depth_scale = sensor.get_depth_scale();
    auto _stream_depth = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto _stream_color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
//    auto _stream_ir = profile.get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
    rs2_intrinsics _intrinsics_depth = _stream_depth.get_intrinsics();
    rs2_intrinsics _intrinsics_color = _stream_color.get_intrinsics();
    cout << "Depth scale = " << _depth_scale << endl;
//    cout<<"color instrinsics: "<<_intrinsics_color<<endl;


    rs2::frameset frames;
    for (int i = 0; i < 10; i++) {
        //Wait for all configured streams to produce a frame
        frames = p.wait_for_frames();
    }

    auto color = frames.get_color_frame();
    auto depth = frames.get_depth_frame();
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();
    Mat depth_image(Size(w, h), CV_16UC1, (void *) depth.get_data(), Mat::AUTO_STEP);
//    cout<<depth_image<<endl;
    int i=0;
    while (true)
    {
        frames = p.wait_for_frames();
        frames = align_to_color.process(frames);
        auto depth = frames.get_depth_frame();
        auto color = frames.get_color_frame();
        Mat color_image(Size(w, h), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);
        Mat depth_image(Size(w, h), CV_16UC1, (void *) depth.get_data(), Mat::AUTO_STEP);
        namedWindow("Display Image", WINDOW_AUTOSIZE);
        imshow("Display Image", depth_image*15);
//        cout << "width: " << image.cols << " height:" << image.rows << " channel: " << image.channels() << endl;
        if(waitKey(15)=='s')
        {
            imwrite("../data/test/rotate_"+to_string(i)+"_depth.png",depth_image);
            imwrite("../data/test/rotate_"+to_string(i)+"_rgb.png",color_image);
            cout<<i<<" saved"<<endl;
            i++;
        }
        if(waitKey(15)=='q')
            break;
    }
    return 0;
}
catch (const rs2::error &e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    "
              << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
