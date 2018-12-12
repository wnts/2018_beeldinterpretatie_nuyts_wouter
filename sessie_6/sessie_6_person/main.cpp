#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/**
 * Classes: HOGDescriptor
                ::setSVMDetector()
                ::detectMultiScale

   - Tracking vector: vector<Point>
   - Resize afbeelding eerst met factor 2 om detector beter te doen werken
   - tracking lijn tekenen door over vector<Point> te loopen en lijn te tekenen
     met line() tss 2 opeenvolgende punten

 */

 const String keys("{help h usage ? |<none>| print this message }"
                  "{@video         |<none>| video test file}");



int main(int argc, char * argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img_test;

    String path_video = parser.get<String>("@video");

    if(path_video.empty())
    {
        cerr << "Please provide all arguments" << endl;
        return 1;
    }

    VideoCapture cap(path_video);
    if(!cap.isOpened())
    {
        fprintf(stderr, "Cannot load video file %s\n", path_video.c_str());
        return 1;
    }

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


    Mat frame;
    vector<Point> track;
    for(;;)
    {
        cap >> frame;
        resize(frame, frame, Size(frame.cols*2, frame.rows*2));
        Mat frame_output = frame.clone();
        vector<Rect> persons;

        hog.detectMultiScale(frame, persons, 0, Size(8,8), Size(32,32), 1.05, 2, false);
        /// Rectangle rond gedetecteerde persoon/personen tekenen
        for (vector<Rect>::iterator i = persons.begin(); i != persons.end(); ++i)
        {
            Rect &r = *i;
            Point track_point(r.tl().x + r.width/2, r.tl().y + r.height/2);
            track.push_back(track_point);
            rectangle(frame_output, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
        }

        /// tracking lijn tekenen
        for(int i = 0; i < track.size(); i++)
        {
            if(i > 0 && norm(track[i]-track[i-1]) < 10)
            {
                line(frame_output, track[i], track[i-1], Scalar(0, 0, 255));
            }
        }
        imshow("persion detection result", frame_output);
        if(waitKey(30) >= 0) break;
    }

    return 0;
}


