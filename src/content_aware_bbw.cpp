//
// Created by parallels on 4/28/18.
//
#include "content_aware_bbw.h"

/*
 * Parameters: Mesh defined by V, F
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LM(Eigen::MatrixXd V, Eigen::MatrixXd F) {
    Eigen::MatrixXd L(F.rows(),F.rows()), M(F.rows(),F.rows());
    for (int i=0; i<F.rows(); i++) {
        Eigen::Vector3d l, cot;
        double r, A;
        l << V.row(F(i,0)) - V.row(F(i,1)), V.row(F(i,1)) - V.row(F(i,2)), V.row(F(i,2)) - V.row(F(i,0));
        r = 0.5*l.sum();
        A = std::sqrt(r*(r-l(0))*(r-l(1))*(r-l(2)));
        cot = (Eigen::Matrix3d::Ones()-2*Eigen::Matrix3d::Identity()) * l.array().square().matrix();
        cot /= A;

        std::vector<int> ijk(F.row(i).data(), F.row(i).data() + F.row(i).size());
        std::vector<int> precomputed_ab = {0,2,0,1,2,1};
        std::vector<int> precomputed_bc = {1,1,2,2,0,0};
        auto index_ab = precomputed_ab.begin();
        auto index_bc = precomputed_bc.begin();
        do{
            L(ijk[0],ijk[1]) -= 0.5 * cot(*index_ab);
            L(ijk[0],ijk[0]) += 0.5 * cot(*index_ab);
            if((cot.array() >= 0).all()) {
                M(ijk[0],ijk[0]) += (1./8.)*cot(*index_ab)*cot(*index_ab)*l(*index_ab);
            } else {
                M(ijk[0],ijk[0]) += (1./8.)*A ? cot(*index_bc) >= 0 : (1./4.)*A;
            }
            index_ab++;
            index_bc++;
        } while (std::next_permutation(ijk.begin(),ijk.end()));
    }
    return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>(L,M);
};

Eigen::MatrixXd bbw(cv::Mat image, cv::Mat roi, int m) {
    std::vector<Eigen::MatrixXd> w(m);
    //First calculate magnitude of gradient of image
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(image, grad_x, CV_32F, 1, 0);
    cv::Sobel(image, grad_y, CV_32F, 0, 1);
    cv::magnitude(grad_x, grad_y, grad_mag);
    //Then calculate gradient of roi==0
    cv::Mat roi_mask, roi_grad;
    roi_mask = roi == 0;
    cv::Sobel(image, grad_x, CV_32F, 1, 0);
    cv::Sobel(image, grad_y, CV_32F, 0, 1);
    cv::magnitude(grad_x, grad_y, roi_grad);



    return Eigen::MatrixXd(0,0);
}

