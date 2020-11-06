#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

#define WINDOW_SIZE 31
#define MAX_DISPARITY 100
#define MIN_DISPARITY 5
#define MARGIN_Y_AXIS 3

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
void reduceVector(vector<double> &v, vector<uchar> status);
bool findCorrWithStereoImg(cv::Mat inLeftImg, cv::Mat inRightImg,std::vector<cv::Point2f> &inLeftCamCorners,
                           std::vector<cv::Point2f> &outRightCorners);
bool LRConsistencyCheck(cv::Mat inLeftImg, cv::Mat inRightImg,std::vector<cv::Point2f> &inLeftCamCorners,
                           std::vector<cv::Point2f> &inRightCorners);
bool extract_stereo_depth(std::vector<cv::Point2f> &srcPoints,std::vector<cv::Point2f> &dstPoints,
                          double baseLine,double focalLength,std::vector<double> &depthVector);
bool transform_depth(std::vector<double> &depthVector,std::vector<cv::Point2f> &srcPoints,cv::Mat proMat,cv::Mat rotMat);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,const cv::Mat &_img2,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts,n_pts_right;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
    ////////////////////////////////////////////////////
    vector<double> stereo_depth;
    vector<double> forw_depth;
    vector<int> matching_table,matching_table2;
    //vector<int> bad_feature_idx;
    double baseLineLength;
    cv::Mat K1,K2,D1,D2,R1,R2,P1,P2,left_cam_mask,right_cam_mask;
    int parallax_idx=0;
    ////////////////////////////////////////////////////
};
