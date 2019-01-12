#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>

using namespace std;
using namespace cv;


int tbThrVal = 70;

void openImgFile(Mat &destination, const String &path)
{
    destination = imread(path);
    if(destination.empty())
    {
        cerr << "Could not open " << path << endl;
        exit(2);
    }
}


int main(int argc, char *argv[])
{
    const String keys("{help h usage ? |<none>| print this message}"
                      "{@img_pcb        |<none>| path to image of PCB}"
                      "{@tpl_dir        |<none>| path to folder containing templates}");
    CommandLineParser cmdParser(argc, argv, keys);
    String pathImgPcb, pathTplDir;
    Mat imgPcb, imgTplC, imgTplR, imgTplL, imgResult;

    pathImgPcb = cmdParser.get<String>("@img_pcb");
    pathTplDir = cmdParser.get<String>("@tpl_dir");

    if(!cmdParser.check())
    {
        cmdParser.printErrors();
        cmdParser.printMessage();
        return 1;
    }

    openImgFile(imgPcb, pathImgPcb);
    openImgFile(imgTplR, pathTplDir + "/R.jpg");
    openImgFile(imgTplC, pathTplDir + "/C.jpg");
    namedWindow("Result");
    createTrackbar("threshold", "Result", &tbThrVal, 100, nullptr);

    while(true)
    {
        imgResult = imgPcb.clone();

        Mat imgTmResultR(imgPcb.rows, imgPcb.cols, imgPcb.type());
        Mat imgTmResultC(imgPcb.rows, imgPcb.cols, imgPcb.type());

        matchTemplate(imgPcb, imgTplR, imgTmResultR, TM_CCORR_NORMED);
        imshow("R template match", imgTmResultR);
        normalize(imgTmResultR, imgTmResultR, 0, 1.0, NORM_MINMAX, CV_32FC1);

        threshold(imgTmResultR, imgTmResultR, tbThrVal / 100.0, 255, THRESH_BINARY);
        imgTmResultR.convertTo(imgTmResultR, CV_8UC1);
        imshow("R template match thresholded", imgTmResultR);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours(imgTmResultR, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        cout << "Found " << contours.size() << " contours" << endl;
        for(const vector<Point> &contour :  contours)
        {
            Point tl = boundingRect(contour).tl();
            rectangle(imgResult, tl, Point(tl.x + imgTplR.cols, tl.y + imgTplR.rows), Scalar(0, 0, 255));
        }

        imshow("Result", imgResult);

        if(waitKey(5) == 'q')
            return 0;
    }
}


