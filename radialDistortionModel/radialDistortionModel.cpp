// radialDistortionModel.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <cmath>
#include <math.h>       /* atan */
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>
#include <stdio.h>


#include "opencv2/core.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <iostream>

using namespace std;
using namespace cv;

const int rownu = 3136;
const int colnu = 4224;

int main() {
    Mat inimg;
    inimg = imread("../grid_v1_edited.jpg", 1);

    cv::Mat distCoeff;
    distCoeff = cv::Mat::zeros(8, 1, 6); // 6:CV_64FC1

    // indices: k1, k2, p1, p2, k3, k4, k5, k6 
    // TODO: add your coefficients here!
    double k1 = -0.0000003;    double k2 = -0.0000000000001;    double p1 = 0;    double p2 = 0;
    double k3 = 0;    double k4 = 0;    double k5 = 0;    double k6 = 0;

    distCoeff.at<double>(0, 0) = k1;    distCoeff.at<double>(1, 0) = k2;    distCoeff.at<double>(2, 0) = p1;    distCoeff.at<double>(3, 0) = p2;
    distCoeff.at<double>(4, 0) = k3;    distCoeff.at<double>(5, 0) = k4;    distCoeff.at<double>(6, 0) = k5;    distCoeff.at<double>(7, 0) = k6;

    //# assume unit matrix for camera
    Mat cam = Mat::eye(3, 3, 5);
    cam.at<float>(0, 2) = colnu / 2.0;  // define center x
    cam.at<float>(1, 2) = rownu / 2.0; // define center y
    cam.at<float>(0, 0) = 10.;        // define focal length x
    cam.at<float>(1, 1) = 10.;        // define focal length y

    //# here the undistortion will be computed
    Mat dst;
    cv::undistort(inimg, dst, cam, distCoeff);
    printf("Part 1 is done!\n");


    Mat camm = Mat::eye(3, 3, 5);
    camm.at<float>(0, 2) = colnu / 2.0;  // define center x
    camm.at<float>(1, 2) = rownu / 2.0; // define center y
    camm.at<float>(0, 0) = 10.;        // define focal length x
    camm.at<float>(1, 1) = 10.;        // define focal length y
    Mat map1, map2;
    initUndistortRectifyMap(camm, distCoeff, Mat(), camm, Size(inimg.cols, inimg.rows), 5, map1, map2);
    printf("Part 2 is done!\n");

    cuda::GpuMat map1GPU(map1);
    cuda::GpuMat map2GPU(map2);
    printf("Part 2 partial is done!\n");
    cuda::GpuMat undis2;
    cuda::GpuMat inimgGPU;
    inimgGPU.upload(inimg);
    cuda::remap(inimgGPU, undis2, map1GPU, map2GPU, 2, 0, cv::Scalar());


    //cuda::GpuMat inimgGPUmapped;
    //cuda::remap(inimgGPU, inimgGPUmapped, m_mapx, m_mapy, cv::INTER_NEAREST, cv::BORDER_WRAP, 0);


    printf("Part 3 is done!\n");


    //cuda::GpuMat inimgGPU;
    
    cuda::GpuMat dstGPU;
    dstGPU.upload(dst);

    cuda::cvtColor(inimgGPU, inimgGPU, 6);
    cuda::cvtColor(dstGPU, dstGPU, 6);
    //cuda::GpuMat inimgGPUmapped;
    //cuda::remap(inimgGPU, inimgGPUmapped, m_mapx, m_mapy, cv::INTER_NEAREST, cv::BORDER_WRAP, 0);
    cuda::GpuMat diffmat;
    cuda::subtract(inimgGPU, dstGPU, diffmat);
    printf("Norm: %d\n", cuda::norm(diffmat, NORM_L2));


    String winname = "Input Image";	namedWindow(winname, WINDOW_OPENGL);
    imshow(winname, inimgGPU);

    String winname2 = "Mapped Image: CPU";	namedWindow(winname2, WINDOW_OPENGL);
    imshow(winname2, dstGPU);

    String winname3 = "Mapped Image: GPU";	namedWindow(winname3, WINDOW_OPENGL);
    imshow(winname3, undis2);
    waitKey(10000);
}