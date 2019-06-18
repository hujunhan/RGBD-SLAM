cmake_minimum_required(VERSION 3.14)
SET(PROJ_NAME "RGBD-SLAM")
project(${PROJ_NAME})
set(ThirdParty "${CMAKE_CURRENT_LIST_DIR}/third-party")
SET(CMAKE_BUILD_TYPE "Release")

# The version number.
set(Tutorial_VERSION_MAJOR 1)
set(Tutorial_VERSION_MINOR 0)

# Using C++ 11
set(CMAKE_CXX_STANDARD 11)


#LIST( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/third-party/g2o/cmake_modules )
#set(G2O_ROOT ${CMAKE_CURRENT_LIST_DIR}/third-party/g2o/g2o)
find_package(Ceres CONFIG REQUIRED)
#find_package(G2O REQUIRED)
#IF(G2O_FOUND)
#    include_directories(${G2O_INCLUDE_DIR})
#    message("G2O lib is found:" ${G2O_INCLUDE_DIR})
#ENDIF(G2O_FOUND)
#message(${G2O_INCLUDE_DIR})

find_package(Eigen3 REQUIRED)
# Using OpenCV and other third-party library
set(INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/helper
        ${CMAKE_CURRENT_LIST_DIR}/include/librealsense2
        ${CMAKE_CURRENT_LIST_DIR}/include/myslam
        ${CMAKE_CURRENT_LIST_DIR}/third-party/glfw-imgui/include
        ${CMAKE_CURRENT_LIST_DIR}/third-party/
        ${CMAKE_CURRENT_LIST_DIR}/third-party/pthread
        ${CMAKE_CURRENT_LIST_DIR}/third-party/opencv-4.1/
        ${EIGEN_INCLUDE_DIR}
        ${CERES_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/third-party/sophus
        ${CMAKE_CURRENT_LIST_DIR}/third-party/ymal-cpp
        ${G2O_INCLUDE_DIRS}/g2o/
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
file(GLOB SOURCE doc/testRtCons.cpp
#        src/myslam/camera.cpp
#        src/myslam/config.cpp
#        src/myslam/frame.cpp
#        src/myslam/map.cpp
#        src/myslam/mappoint.cpp
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
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

target_link_libraries(${PROJ_NAME} ${LINK_DIRS}
        glu32.lib
        opengl32.lib
        ${Sophus_LIBRARIES}
        ${CERES_LIBRARIES}
#        ${G2O_LIBS}
#        myslam
        )
