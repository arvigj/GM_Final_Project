//
// Created by Arvi Gjoka on 5/6/18.
//

#include "mapping.h"

/*
 * w must be numpixels by m
 */
Eigen::MatrixXd mapping(cv::Mat image_s, cv::Mat image_t, Eigen::MatrixXd w, Eigen::MatrixXd T_0) {
    Eigen::MatrixXd G(6*image_s.rows*image_s.cols,1);
    cv::Mat grad_x, grad_y;
    cv::Sobel(image_s, grad_x, CV_32F, 1, 0);
    cv::Sobel(image_s, grad_y, CV_32F, 0, 1);
    std::vector<cv::Mat> C_x, C_y;
    std::vector<Eigen::MatrixXd> C;
    cv::split(grad_x, C_x);
    cv::split(grad_y, C_y);
    for(int i=0; i<C_x.size(); i++) {
        C.push_back(Eigen::MatrixXd(0,0));
        cv::cv2eigen(C_x[i],C[C.size()-1]);
        C.push_back(Eigen::MatrixXd(0,0));
        cv::cv2eigen(C_y[i],C[C.size()-1]);
    }

    //GENERATING J
    Eigen::MatrixXd J;
    J = Eigen::MatrixXd::Zero(6*image_s.rows*image_s.cols, 6*w.cols());
    for(int j=0; j<image_s.cols; j++) {
        for(int i=0; i<image_s.rows; i++) {
            for(int k=0; k<w.cols(); k++) {
                J.block(i+image_s.rows*j,6*k,2,6) <<    j, 0, i, 0, 1, 0,
                                                        0, j, 0, i, 0, 1;
                J.block(i+image_s.rows*j,6*k,2,6) *= w(i+image_s.rows*j,k);
                J.block(2*(image_s.rows*image_s.cols+i+image_s.rows*j),6*k,2,6) << J.block(i+image_s.rows*j,6*k,2,6);
                J.block(2*(2*image_s.rows*image_s.cols+i+image_s.rows*j),6*k,2,6) << J.block(i+image_s.rows*j,6*k,2,6);
            }
        }
    }

    //GENERATING SD
    Eigen::MatrixXd SD(3*image_s.rows*image_s.cols, 6*w.cols());
    for(int i=0; i<3*image_s.rows*image_s.cols; i++) {
        SD.row(i) << G.block(2*i,0,2,1).transpose() * J.block(2*i,0,2,6*w.cols());
    }


    //GENERATING H_i
    int L_half = 6*w.cols()-10;
    std::vector<Eigen::MatrixXd> H_i(std::pow(L_half,2));
    Eigen::MatrixXi grid_x, grid_y;
    grid_x = Eigen::MatrixXi::Zero(image_s.rows, image_s.cols);
    grid_y = Eigen::MatrixXi::Zero(image_s.rows, image_s.cols);
    for(int i=0; i<image_s.rows; i++) {
        grid_y.row(i).array() += i;
    }
    for(int i=0; i<image_s.cols; i++) {
        grid_x.col(i).array() += i;
    }
    Eigen::MatrixXi x(10,10),y(10,10),index(100,1);
    Eigen::MatrixXd intermediate;
    for(int j=0; j<L_half; j++) {
        for(int i=0; i<L_half; i++) {
            x = grid_x.block(i,j,10,10);
            y = grid_y.block(i,j,10,10);
            Eigen::Map<Eigen::VectorXi> cols(x.data(),100);
            Eigen::Map<Eigen::VectorXi> rows(y.data(),100);
            index << rows.eval() + image_s.rows*cols.eval();
            igl::slice(SD, index, 1, intermediate);
            H_i[i+j*L_half] = intermediate.transpose()*intermediate;
        }
    }

    //GENERATING T
    Eigen::MatrixXd T;
    T = T_0;

    //GENERATING A
    Eigen::MatrixXd A(3*image_s.rows*image_s.cols, 2*w.cols());
    std::vector<Eigen::MatrixXd> RGB_mat(3);
    std::vector<Eigen::VectorXd> RGB_vec(3);
    std::vector<cv::Mat> RGB;
    cv::split(image_s, RGB);
    for(int i=0; i<3; i++) {
        cv::cv2eigen(RGB[i], RGB_mat[i]);
        RGB_vec[i] = Eigen::Map<Eigen::VectorXd>(RGB_mat[i].data(), RGB_mat[i].size());
    }
    for(int i=0; i<w.cols(); i++) {
        A.col(2*i) <<   RGB_vec[0].cwiseProduct(w.col(i)),
                        RGB_vec[1].cwiseProduct(w.col(i)),
                        RGB_vec[2].cwiseProduct(w.col(i));
        A.col(2*i+1) << w.col(i), w.col(i), w.col(i);
    }

    //GENERATING Ha_i;
    std::vector<Eigen::MatrixXd> Ha_i(std::pow(L_half,2));
    for(int j=0; j<L_half; j++) {
        for(int i=0; i<L_half; i++) {
            x = grid_x.block(i,j,10,10);
            y = grid_y.block(i,j,10,10);
            Eigen::Map<Eigen::VectorXi> cols(x.data(),100);
            Eigen::Map<Eigen::VectorXi> rows(y.data(),100);
            index << rows.eval() + image_s.rows*cols.eval();
            igl::slice(A, index, 1, intermediate);
            Ha_i[i+j*L_half] = intermediate.transpose()*intermediate;
        }
    }

    Eigen::MatrixXd Z, E, R;
    std::vector<Eigen::MatrixXd> delta_T;
    double epsilon = -1e-4;
    do{


        bool end_flag = true;
        for(auto i=delta_T.begin(); i!=delta_T.end(); i++) {
            if (i->norm() > epsilon) {
                end_flag = false;
            }
        }
        if (end_flag)
            break;
    } while(true);


    return Eigen::MatrixXd(0,0);
}

