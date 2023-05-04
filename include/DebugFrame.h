#ifndef DEBUGFRAME_H
#define DEBUGFRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

class DebugFrame: public Frame
{
public:
    DebugFrame();

    // Copy constructors.
    DebugFrame(const Frame &frame);
    DebugFrame(const DebugFrame &frame);

    // Constructor for Monocular cameras.
    DebugFrame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);
    DebugFrame(const cv::Mat &imGray, const cv::Mat &mask, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    void SetMask(const cv::Mat &mask);

public:
    cv::Mat mmask;
};

} //namespace ORB_SLAM

#endif

