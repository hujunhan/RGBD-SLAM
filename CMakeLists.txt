cmake_minimum_required(VERSION 3.14)
SET(PROJ_NAME "RGBD-SLAM")
project (${PROJ_NAME})

set(ThirdParty "${CMAKE_CURRENT_LIST_DIR}/third-party")


# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# Using C++ 11
set(CMAKE_CXX_STANDARD 11)

# Using OpenCV and other third-party library
set(INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/helper
        ${CMAKE_CURRENT_LIST_DIR}/include/librealsense2
        ${CMAKE_CURRENT_LIST_DIR}/include/myslam
        ${CMAKE_CURRENT_LIST_DIR}/third-party/glfw-imgui/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/
        ${CMAKE_CURRENT_LIST_DIR}/third-party/pthread
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/include/opencv
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/core/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/flann/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/imgproc/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/ml/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/objdetect/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/photo/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/video/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/dnn/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/imgcodecs/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/shape/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/videoio/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/highgui/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/superres/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/ts/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/features2d/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/calib3d/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/stitching/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/videostab/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-3.4/modules/tracking/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/eigen
        ${CMAKE_CURRENT_LIST_DIR}/third-party/sophus

        )


# Link the file
file(GLOB LINK_DIRS ${CMAKE_CURRENT_LIST_DIR}/lib/x64/realsense2.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/x64/glfw-imgui.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_core341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_highgui341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_imgproc341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_calib3d341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_dnn341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_features2d341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_flann341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_imgcodecs341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_ml341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_objdetect341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_photo341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_python2.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_shape341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_stitching341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_superres341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_video341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_videoio341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_videostab341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_tracking341.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/pthreadVC2.lib
        )
# 添加头文件
include_directories(${INCLUDE_DIRS})
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
##设置源文件
file(GLOB SOURCE doc/useEigen.cpp)
# 添加源文件
add_executable(${PROJ_NAME} ${SOURCE} )

#add_executable(RGBD_SLAM main.cpp src/Camera.cpp include/myslam/Camera.h include/myslam/common_include.h)
if ( MSVC )
    set_target_properties( ${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin )
    set_target_properties( ${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_LIST_DIR}/bin )
    set_target_properties( ${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_LIST_DIR}/bin )
endif ( MSVC )
target_link_libraries(${PROJ_NAME} ${LINK_DIRS}
        glu32.lib
        opengl32.lib
        ${Sophus_LIBRARIES}
        )
