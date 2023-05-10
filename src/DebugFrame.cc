#include "DebugFrame.h"
#include "Frame.h"

namespace ORB_SLAM2
{

DebugFrame::DebugFrame()
{}

DebugFrame::DebugFrame(const DebugFrame &frame)    
    : Frame(frame), mmask(frame.mmask)
{}

DebugFrame::DebugFrame(const cv::Mat &imGray, const cv::Mat &mask, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
: Frame(imGray, timeStamp, extractor, voc, K, distCoef, bf, thDepth), mmask(mask)
{}

void DebugFrame::SetMask(const cv::Mat &mask)
{
    mmask = mask;
}

} //namespace ORB_SLAM2