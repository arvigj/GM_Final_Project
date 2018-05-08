//
// Created by Arvi Gjoka on 5/6/18.
//

#ifndef EX5_MAPPING_H
#define EX5_MAPPING_H


#include <Eigen/Core>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <igl/slice.h>


Eigen::MatrixXd mapping(cv::Mat image_s, cv::Mat image_t, Eigen::MatrixXd w, Eigen::MatrixXd T);

#endif //EX5_MAPPING_H
