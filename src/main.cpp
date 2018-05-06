#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "content_aware_bbw.h"

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

    bbw(source, roi, 10);
    //transformations(source, target, roi, Eigen::MatrixXd(1,1));
    if ( !source.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", source);

    waitKey(0);

    return 0;
}
