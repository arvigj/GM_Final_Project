//
// Created by Arvi Gjoka on 5/6/18.
//

#include "mapping.h"

Eigen::MatrixXd mapping(cv::Mat image_s, cv::Mat image_t, Eigen::MatrixXd w, Eigen::MatrixXd T) {
    Eigen::MatrixXd G(6*image_s.rows*image_s.cols);
    return Eigen::MatrixXd(0,0);
}

