//
// Created by parallels on 4/28/18.
//

#ifndef EX5_CONTENT_AWARE_BBW_H
#define EX5_CONTENT_AWARE_BBW_H

#include <Eigen/Core>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <igl/copyleft/marching_cubes.h>
#include <opencv2/core/eigen.hpp>
#include <igl/writeOFF.h>
#include <igl/slice.h>
#include <igl/boundary_loop.h>
//#include <igl/triangle/triangulate.h>

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LM(Eigen::MatrixXd mesh);
Eigen::MatrixXd bbw(cv::Mat image, cv::Mat roi, int m);
cv::Mat gaussian(cv::Size size, cv::Point center, double sigma);


#endif //EX5_CONTENT_AWARE_BBW_H
