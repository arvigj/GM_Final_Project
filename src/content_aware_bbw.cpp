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
    image.convertTo(image, CV_32F);
    roi.convertTo(roi, CV_32F);
    image /= 255;
    roi /= 255;
    cv::Mat image_LAB;
    cv::cvtColor(image, image_LAB, CV_BGR2Lab);
    std::vector<Eigen::MatrixXd> w(m);
    //First calculate magnitude of gradient of image
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(image_LAB, grad_x, CV_32F, 1, 0);
    cv::Sobel(image_LAB, grad_y, CV_32F, 0, 1);
    cv::magnitude(grad_x, grad_y, grad_mag);
    double maxVal;
    cv::minMaxLoc(grad_mag, nullptr, &maxVal, nullptr, nullptr);
    grad_mag /= maxVal;
    //Then calculate gradient of roi==0
    cv::Mat roi_mask, roi_grad;
    roi_mask = roi == 0;
    cv::Sobel(roi_mask, grad_x, CV_32F, 1, 0);
    cv::Sobel(roi_mask, grad_y, CV_32F, 0, 1);
    cv::magnitude(grad_x, grad_y, roi_grad);
    cv::minMaxLoc(roi_grad, nullptr, &maxVal, nullptr, nullptr);
    roi_grad /= maxVal;



    cv::Mat G, G_temp;
    cv::cvtColor(grad_mag, grad_mag, CV_BGR2GRAY);
    G = grad_mag + roi_grad;
    for (int j=0; j<5; j++) {
        cv::GaussianBlur(G, G_temp, cv::Size(27,27), 9, 9);
        G += G_temp;
    }
    cv::minMaxLoc(G, nullptr, &maxVal, nullptr, nullptr);
    G /= maxVal;
    std::cout << maxVal << std::endl;

    cv::Mat C(roi.cols, roi.rows, CV_32F);
    roi.convertTo(roi, CV_32F);
    C = roi.mul(1-G);
    //std::cout << roi << std::endl;
    std::vector<cv::Point> H;
    for (int j=0; j<m; j++) {
        cv::Point maxLoc;
        cv::minMaxLoc(C, nullptr, nullptr, nullptr, &maxLoc);
        H.push_back(maxLoc);
        std::cout << maxLoc << std::endl;


        /*
        cv::Mat i;
        double maxVal;
        cv::minMaxLoc(C, nullptr, &maxVal, nullptr, nullptr);
        cv::divide(C, maxVal/255., i);
        i.convertTo(i, CV_8U);
        cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
        cv::imshow("Display Image", i);
        std::cout << G << std::endl;
        cv::waitKey(0);
         */


        int sigma = std::sqrt(cv::norm(roi > 0, cv::NORM_L2) / (CV_PI*m)) ; //fix sigma here
        std::cout << sigma << std::endl;
        //sigma = 5;
        //std::cout << gaussian(C.size(), maxLoc, sigma) << std::endl;
        /*std::cout << maxLoc << std::endl;
        maxLoc -= cv::Point((int)sigma,(int)sigma);
        //maxLoc.x = maxLoc.x ? maxLoc.x >= 0 : 0;
        //maxLoc.y = maxLoc.y ? maxLoc.y >= 0 : 0;
        std::cout << maxLoc << std::endl;
        cv::Mat gauss_kernel = cv::getGaussianKernel(2*sigma, sigma, CV_32F);
        gauss_kernel.copyTo(Gauss_kernel(cv::Rect(maxLoc, cv::Size(2*sigma, 2*sigma))));
        C -= Gauss_kernel;
         */
        C -= gaussian(C.size(), maxLoc, sigma);

        //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
        //cv::imshow("Display Image", C);
        //cv::waitKey(0);

        cv::circle(image, maxLoc, 3, cv::Scalar(0,255,255),-1);
    }
    /*
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::imshow("ROI", roi);

    cv::waitKey(0);
     */

    for (auto i=H.begin(); i!=H.end(); i++) {
        std::cout << *i << std::endl;
    }
    Eigen::MatrixXd roi_matrix(roi.rows,roi.cols), x_coords(roi.rows,roi.cols), y_coords(roi.rows,roi.cols);
    for (int i=0; i<roi.rows; i++) {
        y_coords.row(i).array() += i;
    }
    for (int i=0; i<roi.cols; i++) {
        x_coords.col(i).array() += i;
    }
    cv::cv2eigen(roi>0, roi_matrix);
    Eigen::Map<Eigen::VectorXd> roi_vec(roi_matrix.data(), roi_matrix.size());
    Eigen::Map<Eigen::VectorXd> x_vec(x_coords.data(), x_coords.size());
    Eigen::Map<Eigen::VectorXd> y_vec(y_coords.data(), y_coords.size());
    Eigen::VectorXd scalar_field(roi_vec.size());
    Eigen::MatrixXd coords(roi_matrix.size(), 3), V;
    Eigen::MatrixXi F;
    scalar_field << roi_vec;
    scalar_field /= 255;
    coords.col(0) << x_vec;
    coords.col(1) << y_vec;
    coords.col(2) << Eigen::VectorXd(roi_matrix.size());
    //igl::copyleft::marching_cubes(scalar_field, coords, roi.cols, roi.rows, 4, V, F);
    //igl::writeOFF("file.off", V, F);
    /*
    Eigen::MatrixXd export_coords;
    std::vector<int> c;
    for (int i=0; i<scalar_field.size(); i++) {
        if (scalar_field(i) == 1) {
            c.push_back(i);
        }
    }
    Eigen::Map<Eigen::VectorXi> cc(c.data(), c.size());
    igl::slice(coords, cc, 1, export_coords);
     */
    /*
    for (int i=0; i<scalar_field.size(); i++) {
        if (scalar_field(i) == 1)
            std::cout << "Row " << coords(i,1) << " Column " << coords(i,0) << " Value " << scalar_field(i) << std::endl;
    }*/
    //std::cout << V << std::endl;
    //igl::writeOFF("file.off",V,F);
    //igl::triangle::triangulate();
    /*
    cv::Mat i;
    double maxVal;
    cv::minMaxLoc(G, nullptr, &maxVal, nullptr, nullptr);
    cv::divide(G, maxVal/255., i);
    i.convertTo(i, CV_8U);
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", i);

    cv::waitKey(0);
     */

    return Eigen::MatrixXd(0,0);
}

cv::Mat gaussian(cv::Size size, cv::Point center, double sigma) {
    cv::Mat gauss(size, CV_32FC1);
    for (int i=0; i<gauss.rows; i++) {
        gauss.row(i) = std::exp(-0.5*std::pow((center.y-i)/sigma, 2));
    }
    for (int i=0; i<gauss.cols; i++) {
        gauss.col(i) *= std::exp(-0.5*std::pow((center.x-i)/sigma, 2));
    }
    //gauss *= 1./(2*CV_PI*sigma*sigma);
    return gauss;
}

Eigen::MatrixXd transformations(cv::Mat image_s, cv::Mat image_t, cv::Mat roi, Eigen::MatrixXd w) {
    assert(roi.channels() == 1);

    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> kp_s, kp_t;
    detector.detect(image_s, kp_s, roi>0);
    detector.detect(image_t, kp_t);
    cv::BFMatcher matcher(cv::NORM_L2);
    cv::SiftDescriptorExtractor extractor;
    cv::Mat d_s, d_t;
    extractor.compute(image_s, kp_s, d_s);
    extractor.compute(image_t, kp_t, d_t);
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;
    matcher.knnMatch(d_s, d_t, matches, 2);
    for (auto i=matches.begin(); i!=matches.end(); i++) {
        cv::DMatch a,b;
        a = i->at(0);
        b = i->at(1);
        if (a.distance < 0.75*b.distance) {
            good_matches.push_back(a);
        }
    }

    auto orient2D = [] (Eigen::Vector2d pa, Eigen::Vector2d pb, Eigen::Vector2d pc) {
        Eigen::Vector3d a,b;
        a << pa - pb, 0;
        b << pc - pb, 0;
        double c = a.cross(b)(2);
        if (c> 0) {
            return 1;
        } else if (c< 0) {
            return -1;
        } else {
            return 0;
        }
    };

    auto incircle = [] (Eigen::Vector2d pa, Eigen::Vector2d pb, Eigen::Vector2d pc, Eigen::Vector2d pd) {

        return 0;
    };

    cv::Mat image_matches;
    cv::drawMatches(image_s, kp_s, image_t, kp_t, good_matches, image_matches);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image_matches);

    cv::waitKey(0);

}
