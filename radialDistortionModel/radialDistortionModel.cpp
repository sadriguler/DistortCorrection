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


    double idealk1, idealk2;
    double maxnorm = 10000000000000000;
    // indices: k1, k2, p1, p2, k3, k4, k5, k6 
    // TODO: add your coefficients here!
    double k1range = 1e-5;
    double k2range = 1e-13;
    double nustage = 100;
    for (double kin1 = 0; kin1 < k1range; ) {
        for (double kin2 = 0; kin2 < k2range; ) {
            double k1 = kin1;    double k2 = kin2;    double p1 = 0;    double p2 = 0;
            double k3 = 0;    double k4 = 0;    double k5 = 0;    double k6 = 0;

            distCoeff.at<double>(0, 0) = k1;    distCoeff.at<double>(1, 0) = k2;    distCoeff.at<double>(2, 0) = p1;    distCoeff.at<double>(3, 0) = p2;
            distCoeff.at<double>(4, 0) = k3;    distCoeff.at<double>(5, 0) = k4;    distCoeff.at<double>(6, 0) = k5;    distCoeff.at<double>(7, 0) = k6;

            //# assume unit matrix for camera
            Mat cam = Mat::eye(3, 3, 5);
            cam.at<float>(0, 2) = colnu / 2.0;  // define center x
            cam.at<float>(1, 2) = rownu / 2.0; // define center y
            cam.at<float>(0, 0) = 1.;        // define focal length x
            cam.at<float>(1, 1) = 1.;        // define focal length y
            // CPU
            // cv::undistort(inimg, dst, cam, distCoeff);

            double x = 4224 / 2;
            double y = 3136 / 2;
            double r = pow(x, 2) + pow(y, 2);
            double ratio1 = (1 + k1 * r + k2 * pow(r, 2));
            printf("Before and after, (x,y): (%f,%f), (%f,%f)\n", x, y, x * ratio1, y * ratio1);
            x = 0;
            r = pow(x, 2) + pow(y, 2);
            double ratio2 = (1 + k1 * r + k2 * pow(r, 2));
            printf("Before and after, (x,y): (%f,%f), (%f,%f)\n", x, y, x * ratio2, y * ratio2);

            printf("Let the game begin; k1: %.16f, k2: %.16f\n", k1, k2);
            printf("ratio1 and ratio2 values: %f %f \n", 1/ratio1, 1/ratio2);

            
            Mat camm = Mat::eye(3, 3, 5);
            camm.at<float>(0, 2) = colnu / 2.0;  // define center x
            camm.at<float>(1, 2) = rownu / 2.0; // define center y
            camm.at<float>(0, 0) = 1.;        // define focal length x
            camm.at<float>(1, 1) = 1.;        // define focal length y
            Mat map1, map2;
            initUndistortRectifyMap(camm, distCoeff, Mat(), camm, Size(inimg.cols, inimg.rows), 5, map1, map2);

            cuda::GpuMat map1GPU(map1);
            cuda::GpuMat map2GPU(map2);
            cuda::GpuMat undis2;
            cuda::GpuMat inimgGPU;
            inimgGPU.upload(inimg);
            cuda::remap(inimgGPU, undis2, map1GPU, map2GPU, 2, 0, cv::Scalar());
            cuda::GpuMat undistResize;
            cuda::resize(undis2, undistResize, Size(0,0),ratio2,ratio2,INTER_LINEAR);

            Mat inimg2;
            inimg2 = imread("../grid_4224_3136.jpg", 1);
            cuda::GpuMat inimgGPU2;
            inimgGPU2.upload(inimg2);


            cuda::cvtColor(inimgGPU2, inimgGPU2, 6);
            cuda::cvtColor(undis2, undis2, 6);
            cuda::GpuMat diffmat;
            cuda::subtract(inimgGPU2, undis2, diffmat);


            double absval = fabs(cuda::norm(diffmat, NORM_L2));
            if (absval < maxnorm){
                maxnorm = absval;
                idealk1 = k1;
                idealk2 = k2;
                printf("Norm: %f\n", maxnorm);
            }

            String winname = "Original Image";	namedWindow(winname, WINDOW_OPENGL);
            imshow(winname, inimgGPU2);

            String winname2 = "undist";	namedWindow(winname2, WINDOW_OPENGL);
            imshow(winname2, undis2);

            String winname4 = "Resized undistorted image";	namedWindow(winname4, WINDOW_OPENGL);
            imshow(winname4, undistResize);

            String winname3 = "Difference";	namedWindow(winname3, WINDOW_OPENGL);
            imshow(winname3, diffmat);
            waitKey(1);
            kin2 += k2range/nustage;
        }
        kin1 += k1range/nustage;
    }
    printf("found values of k1 and k2: %f, %f",idealk1,idealk2);
}