//
// Created by hu on 2019/5/23.
//

#include<windows.h>
#include "librealsense2/rs.hpp"
#include "opencv2/opencv.hpp"
#include "iostream"

using namespace cv;
using namespace std;

int64_t getCurrentTime() {
	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()
		);
	int64_t ret_time = ms.count();
	std::cout << ret_time << std::endl;
	return ret_time;

}
int main(void) try {
    rs2::colorizer color_map;
    rs2::pipeline p;
    rs2::config cfg;
    //使用BRG8才能显示正常的颜色，因为OpenCV就是这样规定显示的
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    auto profile = p.start(cfg);

    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    double _depth_scale = sensor.get_depth_scale();
    auto _stream_depth = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto _stream_color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
//    auto _stream_ir = profile.get_stream(RS2_STREAM_INFRARED).as<rs2::video_stream_profile>();
    rs2_intrinsics _intrinsics_depth = _stream_depth.get_intrinsics();
    rs2_intrinsics _intrinsics_color = _stream_color.get_intrinsics();
    cout << "Depth scale = " << _depth_scale << endl;
//    cout<<"color instrinsics: "<<_intrinsics_color.<<endl;


    rs2::frameset frames;
    for (int i = 0; i < 10; i++) {
        //Wait for all configured streams to produce a frame
        frames = p.wait_for_frames();
    }

    auto color = frames.get_color_frame();
    auto depth = frames.get_depth_frame();

    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();

    //SYSTEMTIME sys;
    int count=0;
    while (true)
    {
        if(++count==100)
            break;
        frames = p.wait_for_frames();
        auto depth = frames.get_depth_frame();
        auto color = frames.get_color_frame();
        Mat color_image(Size(w, h), CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);
        Mat depth_image(Size(w, h), CV_16UC1, (void *) depth.get_data(), Mat::AUTO_STEP);

        //GetLocalTime(&sys);
        stringstream path_depth;
        stringstream path_color;
        stringstream a;
        path_color << "../data/dormitory/rgb/";
        path_depth << "../data/dormitory/depth/" ;
		a << getCurrentTime();
		//a.fill('0');
        //a.width(2); a << sys.wMonth;
        //a.width(2); a << sys.wDay;
        //a.width(2); a << sys.wHour;
        //a.width(2); a << sys.wMinute;
        //a.width(2); a << sys.wSecond;
        //a.width(3); a << sys.wMilliseconds;
        //a << ".png";
        path_color << a.str();
        path_depth << a.str();
        imwrite(path_depth.str(), depth_image);
        imwrite(path_color.str(), color_image);

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
