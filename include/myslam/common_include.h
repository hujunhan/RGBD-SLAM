/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

// define the commonly included file to avoid a long include list
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, 10, 10> Mat1010;
typedef Eigen::Matrix<double, 13, 13> Mat1313;
typedef Eigen::Matrix<double, 8, 10> Mat810;
typedef Eigen::Matrix<double, 8, 3> Mat83;
typedef Eigen::Matrix<double, 6, 6> Mat66;
typedef Eigen::Matrix<double, 5, 3> Mat53;
typedef Eigen::Matrix<double, 4, 3> Mat43;
typedef Eigen::Matrix<double, 4, 2> Mat42;
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 2, 2> Mat22;
typedef Eigen::Matrix<double, 8, 8> Mat88;
typedef Eigen::Matrix<double, 7, 7> Mat77;
typedef Eigen::Matrix<double, 4, 9> Mat49;
typedef Eigen::Matrix<double, 8, 9> Mat89;
typedef Eigen::Matrix<double, 9, 4> Mat94;
typedef Eigen::Matrix<double, 9, 8> Mat98;
typedef Eigen::Matrix<double, 8, 1> Mat81;
typedef Eigen::Matrix<double, 1, 8> Mat18;
typedef Eigen::Matrix<double, 9, 1> Mat91;
typedef Eigen::Matrix<double, 1, 9> Mat19;
typedef Eigen::Matrix<double, 8, 4> Mat84;
typedef Eigen::Matrix<double, 4, 8> Mat48;
typedef Eigen::Matrix<double, 4, 4> Mat44;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix<double, 14, 14> Mat1414;

typedef Eigen::Matrix<double, 14, 1> Vec14;
typedef Eigen::Matrix<double, 13, 1> Vec13;
typedef Eigen::Matrix<double, 10, 1> Vec10;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
// for Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;
// for cv
#include <opencv2/core.hpp>
using cv::Mat;

// for yaml
#include "yaml-cpp/yaml.h"


// std
#include <chrono>
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include "fstream"
#include <set>
#include <unordered_map>
#include <map>
typedef std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> microClock_type;
using namespace std; 
#endif