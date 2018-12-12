#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


const String keys("{help h usage ? |<none>| print this message }"
                  "{@video         |<none>| video test file}"
                  "{@xml_haar      |<none>| classifier file for HAAR}"
                  "{@xml_lbp       |<none>| classifier file for LBP}");


int main(int argc, char * argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img_test;

    String path_video = parser.get<String>("@video");
    String path_xml_haar = parser.get<String>("@xml_haar");
    String path_xml_lbp = parser.get<String>("@xml_lbp");


    if(path_video.empty() or path_xml_haar.empty() or path_xml_lbp.empty())
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


    CascadeClassifier haar(path_xml_haar);
    CascadeClassifier lbp(path_xml_lbp);
    if(haar.empty())
    {
        fprintf(stderr, "Cannot load haar model file %s\n", path_xml_haar.c_str());
        return 1;
    }

    if(lbp.empty())
    {
        fprintf(stderr, "Cannot load lbp model file %s\n", path_xml_lbp.c_str());
        return 1;
    }

    /** Scores berekenen met overloaded detectMultiScale */
    /** Score naast bounding boxes met putText */
    namedWindow("output");
    Mat frame;
    for(;;)
    {
        vector<Rect> faces_haar, faces_lbp;
        cap >> frame;
    	if(frame.empty())
    	{
    		cerr << "Bad or empty frame, skipping this frame..." << endl;
    		continue;
    	}
        Mat frame_output = frame.clone();
        vector<double> weights_haar, weights_lbp;
        vector<int> num_detections_haar, num_detections_lbp;
        haar.detectMultiScale(frame, faces_haar, num_detections_haar);
        lbp.detectMultiScale(frame, faces_lbp, num_detections_lbp);
        if(faces_haar.size() > 0 || faces_lbp.size() > 0)
        {
            char score[8];
            for(size_t i = 0; i < faces_haar.size(); i++)
            {
                sprintf(score, "%d", num_detections_haar[i]);
                Point tp(faces_haar[i].x + faces_haar[i].width, faces_haar[i].y);
                rectangle(frame_output, faces_haar[i], Scalar(0, 255, 0), 3);
                putText(frame_output, score, tp, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0),  1);
            }

            for(size_t i = 0; i < faces_lbp.size(); i++)
            {
                sprintf(score, "%d", num_detections_lbp[i]);
                Point tp(faces_lbp[i].x + faces_lbp[i].width, faces_lbp[i].y + faces_lbp[i].height);
                Point cp(faces_lbp[i].x + faces_lbp[i].width / 2, faces_lbp[i].y + faces_lbp[i].height / 2 );
                circle(frame_output, cp, max(faces_lbp[i].width / 2, faces_lbp[i].height / 2), Scalar(255, 0, 0), 3);
                putText(frame_output, score, tp, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0),  1);

            }
        }
        imshow("output", frame_output);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
