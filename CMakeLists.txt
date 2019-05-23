cmake_minimum_required(VERSION 3.14)
SET(PROJ_NAME "RGBD-SLAM")
project(${PROJ_NAME})

set(ThirdParty "${CMAKE_CURRENT_LIST_DIR}/third-party")


# The version number.
set(Tutorial_VERSION_MAJOR 1)
set(Tutorial_VERSION_MINOR 0)

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
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-4.1/
        ${CMAKE_CURRENT_LIST_DIR}/third-party/eigen
        ${CMAKE_CURRENT_LIST_DIR}/third-party/sophus
        ${CMAKE_CURRENT_LIST_DIR}/third-party/ymal-cpp
        )


# Link the file
file(GLOB LINK_DIRS ${CMAKE_CURRENT_LIST_DIR}/lib/realsense2.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/glfw-imgui.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/opencv_world410.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/pthreadVC2.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/libyaml-cppmd.lib
        )
# 添加头文件
include_directories(${INCLUDE_DIRS})

##设置源文件
file(GLOB SOURCE test/testVO.cpp
        src/myslam/camera.cpp
        src/myslam/config.cpp
        src/myslam/frame.cpp
        src/myslam/map.cpp
        src/myslam/mappoint.cpp
#        src/myslam/visual_odometry.cpp
        )
# 添加源文件
add_executable(${PROJ_NAME} ${SOURCE})

#add_executable(RGBD_SLAM main.cpp src/Camera.cpp include/myslam/Camera.h include/myslam/common_include.h)
if (MSVC)
    set_target_properties(${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)
    set_target_properties(${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_LIST_DIR}/bin)
    set_target_properties(${PROJ_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_LIST_DIR}/bin)
endif (MSVC)
target_link_libraries(${PROJ_NAME} ${LINK_DIRS}
        glu32.lib
        opengl32.lib
        ${Sophus_LIBRARIES}
        #        myslam
        )
