#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<double> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
  //my
  K1=(cv::Mat_<double>(3,3) << 336.959474625917, 0, 323.61860774946,
              0, 337.209187983877, 258.490031281311,
              0.0, 0.0, 1.0);
  K2=(cv::Mat_<double>(3,3) << 336.317792467825,0,316.506898298018,
              0.0, 336.575138530028,260.237169692598,
              0, 0,1);
  D1=(cv::Mat_<double>(1,5) <<-0.349081221612629, 0.157810594010744,-0.000623741459960338,	6.10536825242536e-06,-0.0393820015983333);
  D2=(cv::Mat_<double>(1,5) << -0.347992089492396,	0.154522388325467,-0.000828254293198500,	-0.000272620787288850,-0.0375097680288895);
  R1=(cv::Mat_<double>(3,3) << 0.9997166161895213, -0.009761620289756389, -0.02171170384590232,
      0.00979449507269311, 0.9999510410472643, 0.001408323387455754,
      0.02169689334546639, -0.00162057946774629, 0.9997632812527893);
  R2=(cv::Mat_<double>(3,3) << 0.9998015569097732, -0.001800150422990683, 0.01983951257587237,
      0.00183019795593948, 0.9999972054621756, -0.001496476956558348,
      -0.01983676324997771, 0.001532490226409835, 0.9998020575581299);
  P1=(cv::Mat_<double>(3,4) << 327.839822068036, 0, 320.3756999969482, 0,
      0, 327.839822068036, 258.4395408630371, 0,
      0, 0, 1, 0);
  P2=(cv::Mat_<double>(3,4) << 327.839822068036, 0, 320.3756999969482, -30107.58261836605,
      0, 327.839822068036, 258.4395408630371, 0,
      0, 0, 1, 0);

  cv::Vec3d T(-91.8180341446899,	0.165319079433080,	-1.8219866037574);
  baseLineLength=sqrt(T.dot(T));
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));


    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, pair<int,double>>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], make_pair(ids[i],forw_depth[i]))));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, pair<int,double>>> &a, const pair<int, pair<cv::Point2f, pair<int,double>>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    forw_depth.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            forw_depth.push_back(it.second.second.second);
            ids.push_back(it.second.second.first);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
  for (unsigned int i=0;i<n_pts.size();i++)
  {
      forw_pts.push_back(n_pts.at(i));
      forw_depth.push_back(stereo_depth.at(i));
      ids.push_back(-1);
      track_cnt.push_back(1);
  }
}

void FeatureTracker::readImage(const cv::Mat &_img,const cv::Mat &_img2, double _cur_time)
{
    cv::Mat left_img,right_img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, left_img);
        clahe->apply(_img2, right_img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
    {
        left_img = _img;
        right_img = _img2;
    }

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = left_img;
        left_cam_mask=cv::Mat(left_img.rows, left_img.cols, CV_8UC1, cv::Scalar(0));
        cv::rectangle(left_cam_mask,cv::Point(60,70),cv::Point(570,445),cv::Scalar(255),-1);
    }
    else
    {
        forw_img = left_img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        reduceVector(forw_depth,status);

        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    n_pts.clear();
    n_pts_right.clear();
    matching_table.clear();
    stereo_depth.clear();
    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, left_cam_mask);

            if(n_pts.size()>0)
            {
              cv::Mat map11, map12, map21, map22;
              cv::initUndistortRectifyMap(K1, D1, R1, P1, forw_img.size(), CV_8U, map11, map12);
              cv::initUndistortRectifyMap(K2, D2, R2, P2, right_img.size(), CV_8U, map21, map22);

              cv::Mat img1r, img2r;
              cv::remap(forw_img, img1r, map11, map12, cv::INTER_LINEAR);
              cv::remap(right_img, img2r, map21, map22, cv::INTER_LINEAR);

              std::vector< cv::Point2f > corners1r;
              undistortPoints(n_pts,corners1r,K1,D1,R1,P1);

              if(corners1r.size()>0)
              {
                std::vector<cv::Point2f> corners2r;
                findCorrWithStereoImg(img1r,img2r,corners1r,corners2r);
                //LRConsistencyCheck(img1r,img2r,corners1r,corners2r);
                extract_stereo_depth(corners1r,corners2r,baseLineLength,P1.at<double>(0,0),stereo_depth);
                transform_depth(stereo_depth,corners1r,P1,R1);
              }
              else
              {
                  std::vector<double> tmp_depth(n_pts.size(),-1);
                  stereo_depth=tmp_depth;
              }
            }
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;

        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());

    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        reduceVector(forw_depth,status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

bool findCorrWithStereoImg(cv::Mat inLeftImg, cv::Mat inRightImg,std::vector<cv::Point2f> &inLeftCamCorners,
                           std::vector<cv::Point2f> &outRightCorners)
{
  if(!(inLeftCamCorners.size()>0))
    return false;

  if((inLeftImg.size().width!=inRightImg.size().width) ||(inLeftImg.size().height!=inRightImg.size().height))
    return false;

  outRightCorners.clear();

  std::vector<cv::Point2f> matching_points(inLeftCamCorners.size(),cv::Point2f(-1,-1));

  for(int i=0;i<inLeftCamCorners.size();i++)
  {
    float corner_L_X=inLeftCamCorners.at(i).x;
    float corner_L_Y=inLeftCamCorners.at(i).y;
    int corner_L_X_round=round(corner_L_X);
    int corner_L_Y_round=round(corner_L_Y);

    if((corner_L_X_round<WINDOW_SIZE/2) || (corner_L_Y_round<WINDOW_SIZE/2) || corner_L_X_round>(inLeftImg.size().width-WINDOW_SIZE/2-1) || corner_L_Y_round>(inLeftImg.size().height-WINDOW_SIZE/2-1))
    {
      std::cout<<"rejected corner :  ("<<corner_L_X<<","<<corner_L_Y<<")"<<std::endl;
      continue;
    }
    else
    {
       //std::cout<<inLeftCamCorners.at(i)<<std::endl;

       cv::Rect tempRoi((corner_L_X_round-WINDOW_SIZE/2),(corner_L_Y_round-WINDOW_SIZE/2),WINDOW_SIZE,WINDOW_SIZE);
       cv::Mat tempImg=inLeftImg(tempRoi);

       //disparity 5~100
       int top_left_corner_x=corner_L_X_round-MAX_DISPARITY-WINDOW_SIZE/2;
       if(top_left_corner_x<0)
         top_left_corner_x=0;
       int top_left_corner_y=corner_L_Y_round-WINDOW_SIZE/2-MARGIN_Y_AXIS;
       if(top_left_corner_y<0)
         top_left_corner_y=0;
       cv::Point top_left_corner(top_left_corner_x,top_left_corner_y);

       int bottom_right_corner_x=corner_L_X_round+WINDOW_SIZE/2-MIN_DISPARITY;
       if(bottom_right_corner_x>=inRightImg.size().width)
          bottom_right_corner_x=inRightImg.size().width-1;
       int bottom_right_corner_y=corner_L_Y_round+WINDOW_SIZE/2+MARGIN_Y_AXIS;
       if(bottom_right_corner_y>=inRightImg.size().height)
         bottom_right_corner_y=inRightImg.size().height-1;
       cv::Point bottom_right_corner(bottom_right_corner_x,bottom_right_corner_y);

       cv::Rect imageRoi(top_left_corner,bottom_right_corner);
       cv::Mat inRightImgRoi=inRightImg(imageRoi);

       if(inRightImgRoi.size().width<tempImg.size().width ||inRightImgRoi.size().height<tempRoi.size().height)
       {
         std::cout<<"rejected corner :  ("<<corner_L_X<<","<<corner_L_Y<<")"<<std::endl;
         continue;
       }

       cv::Mat matching_result;
       double maxVal;
       cv::Point maxLoc;

       cv::matchTemplate(inRightImgRoi,tempImg,matching_result,cv::TM_CCOEFF_NORMED);
       cv::minMaxLoc(matching_result,NULL,&maxVal,NULL,&maxLoc);
       //maxLoc.x+=top_left_corner.x+WINDOW_SIZE/2;
       //maxLoc.y+=top_left_corner.y+WINDOW_SIZE/2;
       //std::cout<<cv::Point(maxLoc.x+WINDOW_SIZE/2,maxLoc.y+WINDOW_SIZE/2)<<std::endl;
       cv::Point2f corrPoint(maxLoc.x+top_left_corner_x+WINDOW_SIZE/2,maxLoc.y+top_left_corner_y+WINDOW_SIZE/2);
       matching_points[i]=corrPoint;
       /*
       cv::Mat image_copy;
       inRightImg.copyTo(image_copy);

       //cv::rectangle(image_copy,maxLoc,cv::Point(maxLoc.x+tempImg.cols,maxLoc.y+tempImg.rows),cv::Scalar(255,0,0),2);
       cv::rectangle(image_copy,top_left_corner,bottom_right_corner,cv::Scalar(255,0,0),2);
       cv::circle(image_copy,corrPoint,3,cv::Scalar(255),-1);
       cv::namedWindow("template image",cv::WINDOW_AUTOSIZE);
       cv::imshow("template image",tempImg);

       cv::namedWindow("result image",cv::WINDOW_AUTOSIZE);
       cv::imshow("result image",image_copy);
       cv::waitKey(0);
       */
    }
  }
  outRightCorners=matching_points;
}

bool LRConsistencyCheck(cv::Mat inLeftImg, cv::Mat inRightImg,std::vector<cv::Point2f> &inLeftCamCorners,
                           std::vector<cv::Point2f> &inRightCorners)
{
  if((inLeftCamCorners.empty()) || (inRightCorners.empty()))
    return false;

  if((inLeftImg.size().width!=inRightImg.size().width) ||(inLeftImg.size().height!=inRightImg.size().height))
    return false;

  for(int i=0;i<inLeftCamCorners.size();i++)
  {
    float corner_L_X=inLeftCamCorners.at(i).x;
    float corner_L_Y=inLeftCamCorners.at(i).y;
    int corner_L_X_round=round(corner_L_X);
    int corner_L_Y_round=round(corner_L_Y);

    float corner_R_X=inRightCorners.at(i).x;
    float corner_R_Y=inRightCorners.at(i).y;
    int corner_R_X_round=round(corner_R_X);
    int corner_R_Y_round=round(corner_R_Y);


    if((corner_R_X_round<WINDOW_SIZE/2) || (corner_R_Y_round<WINDOW_SIZE/2) || corner_R_X_round>(inRightImg.size().width-WINDOW_SIZE/2-1) || corner_R_Y_round>(inRightImg.size().height-WINDOW_SIZE/2-1))
    {
      std::cout<<"rejected corner :  ("<<corner_L_X<<","<<corner_L_Y<<")"<<std::endl;
      inRightCorners[i]=cv::Point2f(-1,-1);
      continue;
    }
    else
    {
       //std::cout<<inLeftCamCorners.at(i)<<std::endl;

       cv::Rect tempRoi((corner_R_X_round-WINDOW_SIZE/2),(corner_R_Y_round-WINDOW_SIZE/2),WINDOW_SIZE,WINDOW_SIZE);
       cv::Mat tempImg=inRightImg(tempRoi);

       //disparity 5~100
       int top_left_corner_x=corner_R_X_round+MIN_DISPARITY-WINDOW_SIZE/2;
       if(top_left_corner_x<0)
         top_left_corner_x=0;
       if(top_left_corner_x>=inLeftImg.size().width)
         top_left_corner_x=inLeftImg.size().width-1;
       int top_left_corner_y=corner_R_Y_round-WINDOW_SIZE/2-MARGIN_Y_AXIS;
       if(top_left_corner_y<0)
         top_left_corner_y=0;
       cv::Point top_left_corner(top_left_corner_x,top_left_corner_y);

       int bottom_right_corner_x=corner_R_X_round+WINDOW_SIZE/2+MAX_DISPARITY;
       if(bottom_right_corner_x>=inLeftImg.size().width)
          bottom_right_corner_x=inLeftImg.size().width-1;
       int bottom_right_corner_y=corner_R_Y_round+WINDOW_SIZE/2+MARGIN_Y_AXIS;
       if(bottom_right_corner_y>=inLeftImg.size().height)
         bottom_right_corner_y=inLeftImg.size().height-1;
       cv::Point bottom_right_corner(bottom_right_corner_x,bottom_right_corner_y);

       cv::Rect imageRoi(top_left_corner,bottom_right_corner);
       cv::Mat inLeftImgRoi=inLeftImg(imageRoi);

       if(inLeftImgRoi.size().width<tempImg.size().width ||inLeftImgRoi.size().height<tempRoi.size().height)
       {
         std::cout<<"rejected corner :  ("<<corner_L_X<<","<<corner_L_Y<<")"<<std::endl;
         inRightCorners[i]=cv::Point2f(-1,-1);
         continue;
       }

       cv::Mat matching_result;
       double maxVal;
       cv::Point maxLoc;

       cv::matchTemplate(inLeftImgRoi,tempImg,matching_result,cv::TM_CCOEFF_NORMED);
       cv::minMaxLoc(matching_result,NULL,&maxVal,NULL,&maxLoc);
       //maxLoc.x+=top_left_corner.x+WINDOW_SIZE/2;
       //maxLoc.y+=top_left_corner.y+WINDOW_SIZE/2;
       //std::cout<<cv::Point(maxLoc.x+WINDOW_SIZE/2,maxLoc.y+WINDOW_SIZE/2)<<std::endl;
       cv::Point2f corrPoint(maxLoc.x+top_left_corner_x+WINDOW_SIZE/2,maxLoc.y+top_left_corner_y+WINDOW_SIZE/2);

       float euclidean_distance=sqrt(pow(corner_L_X_round-corrPoint.x,2)+pow(corner_L_Y_round-corrPoint.y,2));
       if(euclidean_distance>3)
       {
         std::cout<<"rejected corner :  ("<<corner_L_X<<","<<corner_L_Y<<")"<<std::endl;
         inRightCorners[i]=cv::Point2f(-1,-1);
         continue;
       }
       /*
       cv::Mat image_copy;
       inRightImg.copyTo(image_copy);

       //cv::rectangle(image_copy,maxLoc,cv::Point(maxLoc.x+tempImg.cols,maxLoc.y+tempImg.rows),cv::Scalar(255,0,0),2);
       cv::rectangle(image_copy,top_left_corner,bottom_right_corner,cv::Scalar(255,0,0),2);
       cv::circle(image_copy,corrPoint,3,cv::Scalar(255),-1);
       cv::namedWindow("template image",cv::WINDOW_AUTOSIZE);
       cv::imshow("template image",tempImg);

       cv::namedWindow("result image",cv::WINDOW_AUTOSIZE);
       cv::imshow("result image",image_copy);
       cv::waitKey(0);
       */
    }
  }
  //outRightCorners=matching_points;
}

bool extract_stereo_depth(std::vector<cv::Point2f> &srcPoints,std::vector<cv::Point2f> &dstPoints,
                          double baseLine,double focalLength,std::vector<double> &depthVector)
{
  if(srcPoints.empty()||dstPoints.empty())
    return false;

  if(srcPoints.size()!=dstPoints.size())
    return false;

  depthVector.clear();

  for(int i=0;i<srcPoints.size();i++)
  {
    float dst_x=dstPoints[i].x;
    float dst_y=dstPoints[i].y;

    if(dst_x<0 ||dst_y<0)
    {
      depthVector.push_back(-1.0);
      continue;
    }
    float src_x=srcPoints[i].x;
    float src_y=srcPoints[i].y;


    //double _disparity=sqrt(pow(src_x-dst_x,2)+pow(src_y-dst_y,2));
    double _disparity=(double)(abs(src_x-dst_x));
    double _depth=baseLine*focalLength/_disparity;
    depthVector.push_back(_depth/1000.0);
  }
}

bool transform_depth(std::vector<double> &depthVector,std::vector<cv::Point2f> &srcPoints,cv::Mat proMat,cv::Mat rotMat)
{
  if(srcPoints.size()!=depthVector.size())
    return false;

  if(depthVector.size()==0)
    return false;

  cv::Mat proMat_inv=proMat(cv::Range(0,3),cv::Range(0,3)).inv();
  cv::Mat rotMat_inv=rotMat.inv();

  for(int i=0;i<depthVector.size();i++)
  {
    double cur_depth=depthVector[i];
    if(!(cur_depth>0))
      continue;

    std::cout<<"before :"<<cur_depth<<std::endl;

    cv::Point2f rect_pt=srcPoints[i];
    cv::Point3f rect_3D_pt(0.0,0.0,0.0);
    double *ptr_proMat=proMat_inv.ptr<double>(0);
    rect_3D_pt.x=ptr_proMat[0]*rect_pt.x+ptr_proMat[1]*rect_pt.y+ptr_proMat[2];
    ptr_proMat=proMat_inv.ptr<double>(1);
    rect_3D_pt.y=ptr_proMat[0]*rect_pt.x+ptr_proMat[1]*rect_pt.y+ptr_proMat[2];
    ptr_proMat=proMat_inv.ptr<double>(2);
    rect_3D_pt.z=ptr_proMat[0]*rect_pt.x+ptr_proMat[1]*rect_pt.y+ptr_proMat[2];

    rect_3D_pt=cur_depth*rect_3D_pt;

    cv::Point3f cam_3D_pt(0.0,0.0,0.0);
    double *ptr_rotMat=rotMat_inv.ptr<double>(0);
    cam_3D_pt.x=ptr_rotMat[0]*rect_3D_pt.x+ptr_rotMat[1]*rect_3D_pt.y+ptr_rotMat[2]*rect_3D_pt.z;
    ptr_rotMat=rotMat_inv.ptr<double>(1);
    cam_3D_pt.y=ptr_rotMat[0]*rect_3D_pt.x+ptr_rotMat[1]*rect_3D_pt.y+ptr_rotMat[2]*rect_3D_pt.z;
    ptr_rotMat=rotMat_inv.ptr<double>(2);
    cam_3D_pt.z=ptr_rotMat[0]*rect_3D_pt.x+ptr_rotMat[1]*rect_3D_pt.y+ptr_rotMat[2]*rect_3D_pt.z;

    std::cout<<"after :"<<cam_3D_pt.z<<std::endl;

    depthVector[i]=cam_3D_pt.z;
  }
}
