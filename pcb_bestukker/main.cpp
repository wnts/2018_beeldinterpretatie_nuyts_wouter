#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <cstdint>

using namespace std;
using namespace cv;


int tbThrTplVal = 70;
int tbThrOutlines = 100;

void openImgFile(Mat &destination, const String &path)
{
    destination = imread(path);
    if(destination.empty())
    {
        cerr << "Could not open " << path << endl;
        exit(2);
    }
}

/**
 *
 * @param imgSearch  Image to search for template
 * @param imgTpl     Template image
 * @param thr        Threshold value for matched templates (must be 0.0 and 1.0 inclusive)
 * @return           Vector of bouding boxes of matched templates
 */
vector<Rect> findTplMatches(Mat &imgSearch, Mat &imgTpl, float thr)
{
    Mat imgTmResult, ccLabels;
    int numComponents = 0;
    vector<Rect> result;

    if(thr > 1.0 || thr < 0.0)
    {
        throw domain_error("Threshold must be between 0.0 and 1.0 inclusive");
    }

    matchTemplate(imgSearch, imgTpl, imgTmResult, TM_CCORR_NORMED);
    normalize(imgTmResult, imgTmResult, 0, 1.0, NORM_MINMAX, CV_32FC1);
    threshold(imgTmResult, imgTmResult, thr, 255, THRESH_BINARY);

    imgTmResult.convertTo(imgTmResult, CV_8UC1); // convert to CV_8UC1 type for connectedComponents()
    numComponents = connectedComponents(imgTmResult, ccLabels, 8);
    for(int i = 1; i < numComponents; i++)
    {
        Point maxLoc;
        Mat imgMaskComponent;
        Mat imgSearch;
        /// maak masker voor de i-de gevonden component
        inRange(ccLabels, i, i, imgMaskComponent);
        /// pas masker toe op match map
        imgSearch = imgTmResult & imgMaskComponent;
        /// nu kunnen we minMaxLoc toepassen op de gemaskte afbeelding die enkel
        /// de ene component bevat
        minMaxLoc(imgSearch, nullptr, nullptr, nullptr, &maxLoc);
        result.emplace_back(Rect(maxLoc, Size(imgTpl.cols, imgTpl.rows)));
    }

    return result;
}

/**
 * Find template matches in an interactive way.
 * The interaction is the displaying of a trackbar so that the user can set the threshold value for the template matching.
 * When the user presses a key, the currently displayed matches are returned.
 * @param imgSearch  Image to search for template
 * @param imgTpl     Template image
 * @param title      String used for the title of the window
 * @param key        Keycode for the key that needs to be pressed by the user to continue
 * @return
 */
vector<Rect> findTplMatchesInteractive(Mat &imgSearch, Mat &imgTpl, int key = 'f', String title = "")
{
    vector<Rect> matches;
    Mat imgResult;
    String windowTitle = "Template matching: " + title;

    int tbTplThr = 75;

    namedWindow(windowTitle);
    createTrackbar("Template matching trackbar", windowTitle, &tbTplThr, 100, nullptr, nullptr);
    while(true)
    {
        imgResult = imgSearch.clone();
        matches = findTplMatches(imgSearch, imgTpl, tbTplThr / 100.0);
        for(const Rect &match : matches)
        {
            rectangle(imgResult, match, Scalar(255, 0, 0));
        }
        imshow(windowTitle, imgResult);
        if(waitKey(5) == key)
            break;
    }
    return matches;
}


int main(int argc, char *argv[])
{
    const String keys("{help h usage ? |<none>| print this message}"
                      "{@img_pcb        |<none>| path to image of PCB}"
                      "{@tpl_dir        |<none>| path to folder containing templates}");
    CommandLineParser cmdParser(argc, argv, keys);
    String pathImgPcb, pathTplDir;
    Mat imgPcb, imgTplC, imgTplR, imgTplROutline, imgTplL;
    Mat imgResult, imgThrOutlines;

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
    openImgFile(imgTplROutline, pathTplDir + "/R_outline.jpg");
    openImgFile(imgTplC, pathTplDir + "/C.jpg");

    /** Template matching on the different components and shapes **/
    vector<Rect> matchesR, matchesROutline, matchesC;
    matchesR = findTplMatchesInteractive(imgPcb, imgTplR, 'f', "Resistors");
    matchesROutline = findTplMatchesInteractive(imgPcb, imgTplROutline, 'f', "Resistors Outline");
    matchesC = findTplMatchesInteractive(imgPcb, imgTplC, 'f', "Capacitors");

    waitKey(0);
}



