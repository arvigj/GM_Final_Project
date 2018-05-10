//
// Created by parallels on 4/28/18.
//

#include "content_aware_bbw.h"


void test_meshing() {
    Eigen::MatrixXi roi_matrix(20,10), x(20,10), y(20,10);
    roi_matrix = Eigen::MatrixXi::Zero(20,10);
    x = Eigen::MatrixXi::Zero(20,10);
    y = Eigen::MatrixXi::Zero(20,10);
    roi_matrix.array() += 1;

    for (int i=0; i<roi_matrix.rows(); i++) {
        y.row(i).array() += i;
    }

    for (int i=0; i<roi_matrix.cols(); i++) {
        x.col(i).array() += i;
    }

    Eigen::Map<Eigen::VectorXi> x_vec(x.data(),x.size());
    Eigen::Map<Eigen::VectorXi> y_vec(y.data(),y.size());

    Eigen::MatrixXi coords(200,3);
    coords.col(0) << x_vec;
    coords.col(1) << y_vec;
    coords.col(2) << Eigen::VectorXi::Zero(x_vec.size());


    std::vector<Eigen::RowVector3i> faces;
    Eigen::Matrix2i small(2,2);
    int index[3];
    for (int j=0; j<roi_matrix.cols()-1; j++) {
        for (int i=0; i<roi_matrix.rows()-1; i++) {
            small = roi_matrix.block(i,j,2,2);
            switch(small.sum()) {
                case 3:
                    if (small(0,0) == 0) {
                        index[0] = (j+1)*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + (i+1);
                        index[2] = (j+1)*roi_matrix.rows() + (i+1);
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else if (small(1,0) == 0) {
                        index[0] = j*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + (i+1);
                        index[2] = (j+1)*roi_matrix.rows() + (i+1);
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else if (small(1,0) == 0) {
                        index[0] = (j+1)*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + i;
                        index[2] = (j+1)*roi_matrix.rows() + (i+1);
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else if (small(1,0) == 0) {
                        index[0] = j*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + (i+1);
                        index[2] = (j+1)*roi_matrix.rows() + i;
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else {
                        std::cout << "SOMETHING IS GOING WRONG WITH MESH GENERATION" << std::endl;
                    }
                    break;
                case 4:
                    index[0] = j*roi_matrix.rows() + i;
                    index[1] = j*roi_matrix.rows() + (i+1);
                    index[2] = (j+1)*roi_matrix.rows() + i;
                    faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    index[0] = (j+1)*roi_matrix.rows() + (i+1);
                    faces.push_back(Eigen::RowVector3i(index[2],index[1],index[0]));
                    break;
                default:
                    break;
            }

        }
    }

    Eigen::MatrixXi F(faces.size(), 3);
    for (int i=0; i<faces.size(); i++) {
        F.row(i) << faces[i];
    }

    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << "Coordinates are " << std::endl;
    std::cout << coords << std::endl;
    std::cout << "Faces are " << std::endl;
    std::cout << F << std::endl;

    //igl::opengl::glfw::Viewer viewer;
    //viewer.data().set_mesh(coords.cast<double>(), F);
    //viewer.core.align_camera_center(coords.cast<double>(), F);
    //viewer.launch();
}


std::pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> LM_(Eigen::MatrixXd V, Eigen::MatrixXi F) {
    Eigen::SparseMatrix<double> L, M;
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
    return std::pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>>(-L,M);
};

/*
 * Parameters: Mesh defined by V, F
 */
std::pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> LM(Eigen::MatrixXd V, Eigen::MatrixXi F) {
    Eigen::SparseMatrix<double> L(V.rows(),V.rows()), M(V.rows(),V.rows());
    std::vector<Eigen::Triplet<double>> L_triplets, M_triplets;
    //First of all need to populate L and M in order to later change the elements of them
    for (int i=0; i<F.rows(); i++) {
        Eigen::VectorXi ijk = F.row(i);
        L_triplets.push_back(Eigen::Triplet<double>(ijk(0),ijk(0),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(1),ijk(1),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(2),ijk(2),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(0),ijk(1),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(0),ijk(2),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(1),ijk(0),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(1),ijk(2),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(2),ijk(0),0.));
        L_triplets.push_back(Eigen::Triplet<double>(ijk(2),ijk(1),0.));
        M_triplets.push_back(Eigen::Triplet<double>(ijk(0),ijk(0),0.));
        M_triplets.push_back(Eigen::Triplet<double>(ijk(1),ijk(1),0.));
        M_triplets.push_back(Eigen::Triplet<double>(ijk(2),ijk(2),0.));
    }
    L.reserve(L_triplets.size());
    M.reserve(M_triplets.size());
    L.setFromTriplets(L_triplets.begin(),L_triplets.end());
    M.setFromTriplets(M_triplets.begin(),M_triplets.end());
    assert(L.sum() == 0);
    assert(M.sum() == 0);
    for (int i=0; i<F.rows(); i++) {
        Eigen::Vector3d l, cot;
        double r, A;
        l << (V.row(F(i,0)) - V.row(F(i,1))).norm(), (V.row(F(i,1)) - V.row(F(i,2))).norm(), (V.row(F(i,2)) - V.row(F(i,0))).norm();
        r = 0.5*l.sum();
        A = std::sqrt(r*(r-l(0))*(r-l(1))*(r-l(2)));
        Eigen::Matrix3d mul;
        mul << -1,1,1,
                1,-1,1,
                1,1,-1;
        cot = mul * l.array().square().matrix();
        cot /= A;
        cot *= .25;

	//std::cout << "l is " << l  << std::endl;
	//std::cout << "cot is " << cot  << std::endl;

        assert(A > 0);

        //std::vector<int> ijk(F.row(i).data(), F.row(i).data() + F.row(i).size());
        Eigen::VectorXi ijk = F.row(i);
        int ab_index, bc_index;
        int count = 0;
        Eigen::MatrixXi index(6,3);
        Eigen::RowVector3i curr;
        index << 0, 1, 2,
                0, 2, 1,
                1, 0, 2,
                1, 2, 0,
                2, 0, 1,
                2, 1, 0;
        do{
            curr << index.row(count++);
		//std::cout << "Triangles are " << ijk(0) << " "<<ijk(1)<<" "<<ijk(2) << std::endl;
		//std::cout << "order of permutation is " << curr << std::endl;
            if ((curr(0) == 0 && curr(1)==1) || (curr(0) == 0 && curr(1)==1)) {
                ab_index = 0;
            } else if ((curr(0) == 0 && curr(1)==2) || (curr(0) == 2 && curr(1)==0)) {
                ab_index = 2;
            } else {
                ab_index = 1;
            }
            if ((curr(1) == 0 && curr(2)==1) || (curr(1) == 0 && curr(2)==1)) {
                bc_index = 0;
            } else if ((curr(1) == 0 && curr(2)==2) || (curr(1) == 2 &&curr(2)==0)) {
                bc_index = 2;
            } else {
                bc_index = 1;
            }
            L.coeffRef(ijk(curr(0)),ijk(curr(1))) += 0.5 * cot(ab_index);
            L.coeffRef(ijk(curr(0)),ijk(curr(0))) -= 0.5 * cot(ab_index);
            if(cot(0)>=0 && cot(1)>=0 && cot(2)>=0) {
                M.coeffRef(ijk(curr(0)),ijk(curr(0))) += (1./8.)*cot(ab_index)*std::pow(l(ab_index),2);
		//std::cout << "Adding "<< M.coeffRef(ijk(curr(0)),ijk(curr(0))) << " to vertex "<<curr(0) << std::endl;
		//std::cout << "AB index is "<<ab_index<<std::endl;
		//std::cout << "cot_ab "<<cot(ab_index)<<std::endl;
		//std::cout << "l_ab "<<l(ab_index)<<std::endl;
            } else {
                if (cot(bc_index) >= 0) {
                    M.coeffRef(ijk(curr(0)),ijk(curr(0))) += (1./8.)*A;
                } else {
                    M.coeffRef(ijk(curr(0)),ijk(curr(0))) += (1./4.)*A;
                }
		//std::cout << "Adding "<< M.coeffRef(ijk(curr(0)),ijk(curr(0))) << " to relative vertex "<<curr(0) <<" with index "<<ijk(curr(0)) << std::endl;
		//std::cout << "AB index is "<<ab_index<<std::endl;
		//std::cout << "A is "<<A<<std::endl;
                }
        } while (count < 6);
	//std::cout << "Repeated " << count << " many times"<<std::endl;
    }
    return std::pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>>(L,M);
};

Eigen::MatrixXd bbw(cv::Mat image, cv::Mat roi, int m) {
    int orig_rows = image.rows;
    double factor = 0.25;
    cv::Mat image_copy;
    image.copyTo(image_copy);
    cv::resize(image,image,cv::Size(), factor,factor,cv::INTER_LINEAR);
    cv::resize(roi,roi,cv::Size(), factor,factor,cv::INTER_LINEAR);
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
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::imshow("ROI", roi);

    cv::waitKey(0);

    Eigen::MatrixXi roi_matrix, x_coords, y_coords;
    x_coords = Eigen::MatrixXi::Zero(roi.rows,roi.cols);
    y_coords = Eigen::MatrixXi::Zero(roi.rows,roi.cols);
    for (int i=0; i<roi.rows; i++) {
        x_coords.row(i).array() += i;
    }
    for (int i=0; i<roi.cols; i++) {
        y_coords.col(i).array() += i;
    }
    cv::cv2eigen(roi>0, roi_matrix);
    roi_matrix /= 255;
    Eigen::Map<Eigen::VectorXi> roi_vec(roi_matrix.data(), roi_matrix.size());
    Eigen::Map<Eigen::VectorXi> x_vec(x_coords.data(), x_coords.size());
    Eigen::Map<Eigen::VectorXi> y_vec(y_coords.data(), y_coords.size());
    Eigen::VectorXi scalar_field(roi_vec.size());
    Eigen::MatrixXi coords(roi_matrix.size(), 3);
    scalar_field << roi_vec;
    coords.col(0) << x_vec;
    coords.col(1) << y_vec;
    coords.col(2) << Eigen::VectorXi::Zero(roi_matrix.size());


    //HERE DO THE MESH TRIANGULATION
    std::vector<Eigen::RowVector3i> faces;
    Eigen::Matrix2i small(2,2);
    int index[3];
    for (int j=0; j<roi.cols-1; j++) {
        for (int i=0; i<roi.rows-1; i++) {
            small = roi_matrix.block(i,j,2,2);
            switch(small.sum()) {
                case 3:
                    if (small(0,0) == 0) {
                        index[0] = (j+1)*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + (i+1);
                        index[2] = (j+1)*roi_matrix.rows() + (i+1);
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else if (small(1,0) == 0) {
                        index[0] = j*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + (i+1);
                        index[2] = (j+1)*roi_matrix.rows() + (i+1);
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else if (small(1,0) == 0) {
                        index[0] = (j+1)*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + i;
                        index[2] = (j+1)*roi_matrix.rows() + (i+1);
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else if (small(1,0) == 0) {
                        index[0] = j*roi_matrix.rows() + i;
                        index[1] = j*roi_matrix.rows() + (i+1);
                        index[2] = (j+1)*roi_matrix.rows() + i;
                        faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    } else {
                        //std::cout << "SOMETHING IS GOING WRONG WITH MESH GENERATION" << std::endl;
                    }
                    break;
                case 4:
                    index[0] = j*roi_matrix.rows() + i;
                    index[1] = j*roi_matrix.rows() + (i+1);
                    index[2] = (j+1)*roi_matrix.rows() + i;
                    faces.push_back(Eigen::RowVector3i(index[0],index[1],index[2]));
                    index[0] = (j+1)*roi_matrix.rows() + (i+1);
                    faces.push_back(Eigen::RowVector3i(index[2],index[1],index[0]));
                    break;
                default:
                    break;
            }

        }
    }

    Eigen::MatrixXi F_(faces.size(), 3);
    for (int i=0; i<faces.size(); i++) {
        F_.row(i) << faces[i];
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F, IM;
    igl::remove_unreferenced(coords.cast<double>(), F_, V, F, IM);


    std::vector<cv::Mat> channels;
    Eigen::MatrixXi L, A, B;
    cv::split(image_LAB, channels);
    cv::cv2eigen(channels[0], L);
    cv::cv2eigen(channels[1], A);
    cv::cv2eigen(channels[2], B);
    Eigen::Map<Eigen::VectorXi> L_vec(L.data(),L.size());
    Eigen::Map<Eigen::VectorXi> A_vec(A.data(),A.size());
    Eigen::Map<Eigen::VectorXi> B_vec(B.data(),B.size());
    Eigen::VectorXi L_vec_incl, A_vec_incl, B_vec_incl;
    igl::slice(L_vec, V.col(0).cast<int>() + image_LAB.rows*V.col(1).cast<int>(), 1, L_vec_incl);
    igl::slice(A_vec, V.col(0).cast<int>() + image_LAB.rows*V.col(1).cast<int>(), 1, A_vec_incl);
    igl::slice(B_vec, V.col(0).cast<int>() + image_LAB.rows*V.col(1).cast<int>(), 1, B_vec_incl);
    std::cout << V.rows() << std::endl;
    std::cout << L_vec_incl.size() << std::endl;
    std::cout << A_vec_incl.size() << std::endl;
    std::cout << B_vec_incl.size() << std::endl;
    V.conservativeResize(V.rows(), 5);
    V.col(2) << 16. * L_vec_incl.cast<double>()/255.;
    V.col(3) << 16. * A_vec_incl.cast<double>()/255.;
    V.col(4) << 16. * B_vec_incl.cast<double>()/255.;

    /*
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V.block(0,0,V.rows(),3), F);
    viewer.core.align_camera_center(V.block(0,0,V.rows(),3), F);
    viewer.launch();
     */

    auto lm = LM_(V,F);
    Eigen::SparseMatrix<double> Q(V.rows(),V.rows());
    //Check if positive semi definite
    /*
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> lltOfA(Q);
    if (lltOfA.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Possibly non PSD matrix");
    }
     */

    for(int k=0; k<lm.second.outerSize(); k++) {
        for(Eigen::SparseMatrix<double>::InnerIterator it(lm.second,k); it; ++it) {
            if (lm.second.coeffRef(it.row(), it.col()) == 0) {
                std::cout << "Zero in mass matrix" << std::endl;
            }
        }
    }

    Q = lm.first * lm.second.diagonal().asDiagonal().inverse() * lm.first;


    std::vector<Eigen::Triplet<double >> w_h;
    std::vector<Eigen::SparseMatrix<double>> constraints;
    for (auto i=H.begin(); i!=H.end(); i++) {
        //figure out what index to put in here
        int old_size = image.rows*i->x + i->y;
        if (IM(old_size,0) == -1) {
            H.erase(i);
            continue;
        } else {
            old_size = IM(old_size,0);
        }
        w_h.push_back(Eigen::Triplet<double>(0,old_size,1.));
        constraints.push_back(Eigen::SparseMatrix<double>(1,Q.rows()));
        constraints[constraints.size()-1].reserve(1);
        constraints[constraints.size()-1].setFromTriplets(w_h.begin(), w_h.end());
        w_h.pop_back();
    }
    Eigen::MatrixXd W(Q.rows(), constraints.size());
    igl::mosek::MosekData mosek_data;
    //TODO try solving this multiple times for each handle
	std::cout << "Reached mosek" << std::endl;
	for (int i=0; i<constraints.size(); i++) {
	    Eigen::VectorXd solution;
        bool converged = igl::mosek::mosek_quadprog(Q, Eigen::VectorXd::Zero(Q.rows()), 0, constraints[i], Eigen::VectorXd::Ones(1).array()-0.1, Eigen::VectorXd::Ones(1), Eigen::VectorXd::Zero(Q.rows()), Eigen::VectorXd::Ones(Q.rows()), mosek_data, solution);
        W.block(0,i,W.rows(),1) << solution;
        std::cout << "Convergence: " << converged << std::endl;
	}
	igl::normalize_row_sums(W,W);
    Eigen::MatrixXd Col;
    for(int i=0; i<W.cols(); i++) {
        igl::jet(W.col(i), true, Col);
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V.block(0,0,V.rows(),3), F);
        viewer.core.align_camera_center(V.block(0,0,V.rows(),3), F);
        viewer.data().set_colors(Col);
        viewer.launch();
    }

    Eigen::MatrixXd W_out = Eigen::MatrixXd::Zero(image.rows*image.cols, W.cols());
    for(int i=0; i<W.rows(); i++) {
        W_out.row(V(i,1)*image.rows+V(i,0)) = W.row(i);
    }
    /*
    for (int i=0; i<W_out.rows(); i++) {
        if (IM(i,0) != -1) {
            W_out.row(i) << W.row(IM(i,0));
        }

    }
     */
    Eigen::MatrixXd W_upscaled(image_copy.rows*image_copy.cols,W_out.cols());
    for (int i=0; i<W_out.cols(); i++) {
        Eigen::VectorXd w_channel = W_out.col(i);
        Eigen::MatrixXd w_matrix = Eigen::Map<Eigen::MatrixXd, Eigen::ColMajor>(w_channel.data(), image.rows, image.cols);
        /*
        Eigen::MatrixXd w_matrix(image.rows, image.cols);
        for(int j=0; j<w_matrix.rows(); j++) {
            w_matrix.row(j) = w_channel.block(j*image.cols, 0, image.cols, 1).array().transpose();
        }
         */
        cv::Mat w_up;
        cv::eigen2cv(w_matrix, w_up);
        cv::resize(w_up, w_up, cv::Size(image_copy.rows, image_copy.cols), cv::INTER_CUBIC);
        cv::cv2eigen(w_up, w_matrix);
        W_upscaled.col(i) << Eigen::Map<Eigen::VectorXd>(w_matrix.data(), w_matrix.size());
    }

    Eigen::VectorXd weights = W_out.col(1);
    Eigen::MatrixXd alpha = Eigen::Map<Eigen::MatrixXd>(weights.data(), image.rows, image.cols);
    alpha *= 255;
    cv::Mat a;
    cv::eigen2cv(alpha, a);
    /*
    std::vector<cv::Mat> RGB_channels;
    cv::split(image_copy,RGB_channels);
    RGB_channels.push_back(a);
    for (auto i=RGB_channels.begin(); i!=RGB_channels.end(); i++) {
        std::cout << i->rows << "\t" << i->cols << "\t" << i->depth() << std::endl;
    }
    cv::merge(RGB_channels, image_new);
    image_new.convertTo(image_new, CV_8UC4);
     */

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Image", a);
    cv::waitKey(0);
    //igl::mosek::mosek_quadprog(Q, Eigen::VectorXd::Zero(Q.rows()), 0, constraints, Eigen::VectorXd::Ones(1), Eigen::VectorXd::Ones(1), Eigen::VectorXd::Zero(Q.rows()), Eigen::VectorXd::Ones(Q.rows()), mosek_data, v);

	/*
    std::vector<int> loops;
    igl::boundary_loop(F, loops);
    std::sort(loops.begin(), loops.end());
    std::vector<int> nonzero(V.rows());
    Eigen::MatrixXi all_faces = Eigen::VectorXi::LinSpaced(V.rows(),0,V.rows()-1);
    std::vector<int> all(all_faces.data(), all_faces.data()+all_faces.size());
    std::vector<int>::iterator it;
    it = std::set_difference(all.begin(),all.end(),loops.begin(),loops.end(),nonzero.begin());
    nonzero.resize(it-nonzero.begin());
    Eigen::Map<Eigen::VectorXi> valid(nonzero.data(), nonzero.size());
    std::cout << V.rows() - valid.rows() << std::endl;

    Eigen::SparseMatrix<double> L_, M_;
    Eigen::SparseMatrix<double> Q_small;
    igl::slice(lm.first, valid, valid, L_);
    igl::slice(lm.second, valid, valid, M_);
    Q_small = L_ * M_.diagonal().asDiagonal().inverse() * L_;
    igl::mosek::mosek_quadprog(Q_small, Eigen::VectorXd::Zero(Q_small.rows()), 0, constraints, Eigen::VectorXd::Ones(5), Eigen::VectorXd::Ones(5), Eigen::VectorXd::Zero(Q_small.rows()), Eigen::VectorXd::Ones(Q_small.rows()), mosek_data, v);
*/

    //igl::opengl::glfw::Viewer viewer;
    //viewer.data().set_mesh(coords.block(0,0,coords.rows(),3).cast<double>(), F);
    //viewer.core.align_camera_center(coords.block(0,0,coords.rows(),3).cast<double>(), F);
    //viewer.launch();
    //igl::writeOFF("file.off",coords,F);


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

    return W_upscaled;
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
    std::vector<cv::Point2d> M_s, M_t;
    matcher.knnMatch(d_s, d_t, matches, 2);
    Eigen::MatrixXi matches_duplicate = Eigen::MatrixXi::Zero(image_s.rows, image_s.cols);
    matches_duplicate.array() -= 1;
    for (auto i=matches.begin(); i!=matches.end(); i++) {
        cv::DMatch a,b;
        a = i->at(0);
        b = i->at(1);
        if (a.distance < 0.75*b.distance) {
            cv::Point s, t;
            s = kp_s[a.queryIdx].pt;
            t = kp_t[a.trainIdx].pt;
            if (matches_duplicate(s.y, s.x) >= 0) {
                if (a.distance < good_matches[matches_duplicate(s.y, s.x)].distance) {
                    M_s[matches_duplicate(s.y,s.x)] = s;
                    M_t[matches_duplicate(s.y,s.x)] = t;
                    good_matches[matches_duplicate(s.y,s.x)] = a;
                }
            } else {
                M_s.push_back(s);
                M_t.push_back(t);
                matches_duplicate(s.y, s.x) = (int)good_matches.size();
                good_matches.push_back(a);
            }
        }
    }

    typedef Eigen::MatrixXd::Scalar Scalar;

    //auto orient2D = [] (Eigen::Scalar pa[2], Eigen::Scalar pb[2], Eigen::Scalar pc[2]) {
    auto orient2D = [] (const Scalar pa[2], const Scalar pb[2], const Scalar pc[2]) {
        std::cout << pa[0] << "\t" << pa[1] << std::endl;
        std::cout << pb[0] << "\t" << pb[1] << std::endl;
        std::cout << pc[0] << "\t" << pc[1] << std::endl;
        std::cout << std::endl;
        Eigen::Vector3d a,b;
        a << pa[0] - pb[0], pa[1] - pb[1], 0;
        b << pc[0] - pb[0], pc[1] - pb[1], 0;
        double c = a.cross(b)(2);
        if (c> 0) {
            return 1;
        } else if (c< 0) {
            return -1;
        } else {
            return 0;
        }
    };

    //auto incircle = [] (Eigen::Scalar pa[2], Eigen::Scalar pb[2], Eigen::Scalar pc[2], Eigen::Scalar pd[2]) {
    auto incircle = [] (const Scalar pa[2], const Scalar pb[2], const Scalar pc[2], const Scalar pd[2]) {
        Eigen::Matrix4d mat;
        mat.col(0) << std::sqrt(pd[0]*pd[0]+pd[1]*pd[1]), std::sqrt(pa[0]*pa[0]+pa[1]*pa[1]),std::sqrt(pb[0]*pb[0]+pb[1]*pb[1]),std::sqrt(pc[0]*pc[0]+pc[1]*pc[1]);
        mat.col(1) << pd[0], pa[0], pb[0], pc[0];
        mat.col(2) << pd[1], pa[1], pb[1], pc[1];
        mat.col(3) << 1,1,1,1;
        double d = mat.determinant();
        if (d > 0) {
            return -1;
        } else if (d < 0) {
            return 1;
        } else {
            return 0;
        }
    };

    Eigen::MatrixXd V_S(M_s.size(),2), V_T(M_t.size(),2);
    assert(V_S.rows() == V_T.rows());
    for (int i=0; i<V_S.rows(); i++) {
        V_S.row(i) << M_s[i].x, M_s[i].y;
        V_T.row(i) << M_t[i].x, M_t[i].y;
    }
    //std::cout << "V_S: " << V_S << std::endl;
    //std::cout << "V_T: " << V_T << std::endl;
    //TODO FIX DELAUNAY TRIANGULATION FOR DUPLICATE POINTS
	Eigen::MatrixXi  F;
    igl::copyleft::cgal::delaunay_triangulation(V_S, F);
    //igl::copyleft::cgal::delaunay_triangulation(V_S, F_);
    //return Eigen::MatrixXd(0,0);
    //Eigen::MatrixXd V_(V_S.rows(),3);
    //V_.block(0,0,V_S.rows(),2) = V_S;
    //igl::writeOFF("/home/parallels/Desktop/Parallels Shared Folders/Downloads/file.off", V_, F);
	//return Eigen::MatrixXd(0,0);
    /*
    Eigen::MatrixXi F(65,3);
    F << 24, 19, 12,
            19, 18, 12,
            15, 16, 14,
            19, 17, 18,
            34, 33, 35,
            30, 21, 12,
            9, 30, 10,
            18, 13, 12,
            13, 30, 12,
            30, 13, 10,
            37, 17, 19,
            17, 37, 15,
            15, 37, 16,
            37, 19, 24,
            23, 34, 35,
            1, 3, 25,
            33, 28, 35,
            28, 22, 35,
            22, 28, 4,
            3, 20, 25,
            20, 32, 25,
            32, 31, 4,
            31, 20, 3,
            20, 31, 32,
            11, 9, 10,
            11, 23, 35,
            13, 11, 10,
            11, 13, 18,
            23, 11, 18,
            30, 29, 21,
            9, 29, 30,
            23, 7, 34,
            17, 7, 18,
            7, 23, 18,
            7, 15, 14,
            7, 17, 15,
            21, 2, 0,
            2, 1, 0,
            27, 32, 4,
            28, 27, 4,
            32, 27, 33,
            27, 28, 33,
            5, 29, 9,
            11, 5, 9,
            5, 26, 21,
            29, 5, 21,
            5, 11, 35,
            26, 5, 35,
            7, 6, 34,
            34, 6, 33,
            6, 32, 33,
            6, 7, 14,
            25, 6, 14,
            32, 6, 25,
            36, 2, 21,
            36, 22, 4,
            31, 36, 4,
            36, 31, 3,
            1, 36, 3,
            2, 36, 1,
            26, 8, 21,
            8, 36, 21,
            36, 8, 22,
            22, 8, 35,
            8, 26, 35;
     */
    /*
    cv::Mat image_matches;
    cv::drawMatches(image_s, kp_s, image_t, kp_t, good_matches, image_matches);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image_matches);

    cv::waitKey(0);
     */
    std::vector<int> valid_triangles;
    for (int i=0; i<F.rows(); i++) {
        Eigen::RowVector3d a,b;
        a << V_T.row(F(i,0)) - V_T.row(F(i,1)) , 0;
        b << V_T.row(F(i,2)) - V_T.row(F(i,1)) , 0;
        if (a.cross(b)(2) > 0) {
            valid_triangles.push_back(i);
        }
    }
    Eigen::Map<Eigen::VectorXi> valid_t(valid_triangles.data(), valid_triangles.size());
    Eigen::MatrixXi F_valid;
    igl::slice(F, valid_t, 1, F_valid);

    std::cout << F_valid << std::endl;
    std::vector<Eigen::MatrixXd> piecewise_affine;
    for (auto i=0; i<F_valid.rows(); i++) {
        Eigen::MatrixXd t(3,3), t_(2,3);
        Eigen::VectorXd a(3), b(2);
        for (int j=0; j<3; j++) {
            a << V_S(F_valid(i,j),0), V_S(F_valid(i,j),1), 1;
            b << V_T(F_valid(i,j),0), V_T(F_valid(i,j),1);
            t.col(j) = a;
            t_.col(j) = b;
        }
        piecewise_affine.push_back(t_ * t.inverse());
    }

    cv::Mat A_match;
    Eigen::MatrixXd A_eig;
    A_match = cv::findHomography(M_s, M_t, CV_RANSAC);
    cv::cv2eigen(A_match, A_eig);
    std::cout << A_match << std::endl;

    //BUILD THE LINEAR SYSTEM TO SOLVE
    auto intriangle = [](Eigen::RowVector3d a, Eigen::RowVector3d b, Eigen::RowVector3d c) {
        double base = a.cross(b)(2);
        double s = c.cross(b)(2) / base;
        double t = a.cross(c)(2) / base;
        if((s>=0) && (t>=0) && (s+t<=1))
            return 1;
        return 0;
    };
    double gamma = 0.1;
    std::vector<std::vector<int>> vertices;
    int tot = 0;
    for(int i=0; i<F_valid.rows(); i++) {
        vertices.push_back(std::vector<int>());
        //RASTERIZE EACH TRIANGLE USING BARYCENTRIC APPROACH
        Eigen::MatrixXd v;
        igl::slice(V_S, F_valid.row(i), 1, v);
        Eigen::Vector2d minv = v.colwise().minCoeff();
        Eigen::Vector2d maxv = v.colwise().maxCoeff();
        Eigen::RowVector3d a(0,0,0), b(0,0,0), c(0,0,0);
        a << v.row(1) - v.row(0), 0;
        b << v.row(2) - v.row(0), 0;
        for(int x=minv(0); x<maxv(0); x++) {
            for(int y=minv(1); y<maxv(1); y++) {
                c << x, y, 0;
                c.block(0,0,1,2) -= v.row(0);
                if (intriangle(a,b,c)) {
                    vertices[i].push_back(y+x*image_s.rows);
                    tot++;
                }
            }
        }

    }
    //std::cout << "These are the vertices" << std::endl;
    //for(int i=0; i<vertices.size(); i++) {
    //    std::cout << vertices[i].size() << std::endl;
    //}
    Eigen::MatrixXd w_T(tot,w.cols()), M;
    M = Eigen::MatrixXd::Zero(tot,6);
    int index = 0;
    Eigen::MatrixXd A_ = Eigen::MatrixXd::Zero(6,6);
    Eigen::MatrixXd b_ = Eigen::MatrixXd::Zero(6,w.cols());
    Eigen::MatrixXd p_, ptp, ptMp, W(w.cols(),1);
    for(int i=0; i<F_valid.rows(); i++) {
        for(int p=0; p<vertices[i].size(); p++) {
            W << w.row(vertices[i][p]).array().transpose();
            p_ = Eigen::KroneckerProduct<Eigen::MatrixXd, Eigen::MatrixXd>(Eigen::Vector3d(vertices[i][p]%image_s.rows,vertices[i][p]%image_s.rows,1),Eigen::MatrixXd::Identity(2,2)).transpose();
            ptp = p_.transpose() * p_;
            ptMp = p_.transpose() * piecewise_affine[i] * Eigen::Vector3d(vertices[i][p]%image_s.rows,vertices[i][p]%image_s.rows,1);
            ptMp *= W.transpose();
            //ptMp /= W.squaredNorm();
            A_ += (double)roi.at<int>(vertices[i][p]/image_s.rows, vertices[i][p]%image_s.rows)/255. * ptp;
            b_ += (double)roi.at<int>(vertices[i][p]/image_s.rows, vertices[i][p]%image_s.rows)/255. * ptMp;
        }
    }

    double lambda = 0.1;
    for(int i=0; i<image_s.rows*image_s.cols; i++) {
        double r = roi.at<int>(i/image_s.rows, i%image_s.rows)/255.;
        if (r == 0) continue;
        W << w.row(i).array().transpose();
        p_ = Eigen::KroneckerProduct<Eigen::MatrixXd, Eigen::MatrixXd>(Eigen::Vector3d(i%image_s.rows,i%image_s.rows,1),Eigen::MatrixXd::Identity(2,2)).transpose();
        ptp = p_.transpose() * p_;
        ptMp = p_.transpose() * A_eig.block(0,0,2,3) * Eigen::Vector3d(i%image_s.rows,i%image_s.rows,1);
        ptMp *= W.transpose();
        //ptMp /= W.squaredNorm();
        A_ += r*lambda*ptp;
        b_ += r*lambda*ptMp;
    }

    Eigen::MatrixXd x = A_.householderQr().solve(b_);
    std::cout << "A matrix: " << A_ << std::endl;
    std::cout << "b matrix: " << b_ << std::endl;
    std::cout << "x matrix: " << x << std::endl;
    return x;
}

