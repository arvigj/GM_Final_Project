#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "content_aware_bbw.h"
#include "mapping.h"

using namespace cv;

int main(int argc, char** argv) {
    if ( argc < 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }


    Mat source, target, roi;
    source = imread( argv[1], CV_LOAD_IMAGE_COLOR );
    target = imread( argv[3], CV_LOAD_IMAGE_COLOR );
    roi = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

    std::cout << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
    //bbw(source, roi, 10);
    transformations(source, target, roi, Eigen::MatrixXd(source.rows*source.cols,5));
    //mapping(source, target, Eigen::MatrixXd(source.rows*source.cols,5), Eigen::MatrixXd(1,6));
    //test_meshing();
    /*
    if ( !source.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", source);

    waitKey(0);
     */

    return 0;
}
