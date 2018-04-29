//
// Created by parallels on 4/28/18.
//

#ifndef EX5_CONTENT_AWARE_BBW_H
#define EX5_CONTENT_AWARE_BBW_H

#include <Eigen/Core>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LM(Eigen::MatrixXd mesh);
Eigen::MatrixXd bbw();


#endif //EX5_CONTENT_AWARE_BBW_H
