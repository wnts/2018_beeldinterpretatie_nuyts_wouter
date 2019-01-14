/**
 * Program for PCB assembly with computer vision techniques.
 * Given a picture of an unassembled PCB, detect the various outlines and component designators, and virtually "assemble" the components on the PCB
 * With the limited time available for this project, it is not very robust, since it relies mostly on template matching and requires user assistance for proper operation.
 *
 * Usage: /pcb_bestukker/input/pcb.jpg ../pcb_bestukker/input
 * - Then you are presented with a window including a trackbar for selecting the 'R' designators
 *   After setting the trackbar to a suitable value, press 'n' to continue to the next window
 *   Three more windows for matching the 'R' outlines, and 'C', and 'D' designators follow.
 * - Next, a window with a grayscale image appears. You should select a suitable value for the trackbar such that the PCB holes dissappear,
 *   while at the same time the outlines of the components appear as intact/closed lines.
 *   They are segmented and removed by creating a mask as follows:
 *      - A binary image is created by thresholding the grasycale PCB image using the value set by the trackbar.
 *      - The holes are semgented by applying erosion and dilation to the binary image.
 *      - The resulting binary image containg only the holes is used as a mask for the original image.
 * - Next, a window with the original PCB image and a trackbar appears.
 *   The trackbar should be set to a value such that the outlines of the components (except for resistors and ICs) are surrounded with a rectangle.
 *   This step uses the binary image from the previous window, with the holes segmented out. Connected components analysis is applied to obtain the various outlines.
 *   Only large connected components represent the outlines, so the trackbar is used to select a value for the minimum pixel area of the connected components to keep.
 * - Finally, the resulted "assembled" PCB appears.
 *   Assembly is done by matching the previously segmented designators and outlines based on distance.
 *
 *
 *
 *
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <cstdint>
#include <utility>
#include <math.h>
#include <limits>

using namespace std;
using namespace cv;


int tbThrTplVal = 70;

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
 * Helper function for getting the (approximate) center pixel of a rectangle
 * @param rect  The rectangle to get the center of
 * @return      A point indicating the location of the center pixel
 */
Point getRectCenter(Rect rect)
{
    return Point(rect.tl().x + rect.width / 2, rect.tl().y + rect.height / 2);
}

/**
 * Helper function for computing the pixel distance between two points
 * The distance computed is the Euclidian distance.
 * @param p1 First point
 * @param p2 Second point
 * @return Euclidian pixel distance between p1 and p2
 */
float getPixelDistance(Point p1, Point p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// Return the rotation matrices for each rotation
// The angle parameter is expressed in degrees!
//void rotate(Mat& src, double angle, Mat& dst)
//{
//    Point2f pt(src.cols/2., src.rows/2.);
//    Mat r = getRotationMatrix2D(pt, angle, 1.0);
//    warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
//}

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
    vector<Mat> imgTmResults;
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
 * When the user presses the specified key, the currently displayed matches are returned.
 * @param imgSearch  Image to search for template
 * @param imgTpl     Template image
 * @param title      String used for the title of the window
 * @param key        Keycode for the key that needs to be pressed by the user to continue
 * @return
 */
vector<Rect> findTplMatchesInteractive(Mat &imgSearch, Mat &imgTpl, int key = 'n', String title = "")
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

/**
 * Helper function for creating pairs of component outlines and the nearest designator
 * The rectangles in outlines are matched to the rectangles in designators by finding the closest matching designator for each outline
 * @param outlines(designators)
 * @param designators(outlines)
 * @return A vector of pairs, with the first element of the pair the outline(designator), and the second element the closest matched component designator(outline)
 */
vector<pair<Rect, Rect> > getDesignatorOutlinePairs(vector<Rect> outlines, vector<Rect> designators)
{
    vector<pair<Rect, Rect> > pairs; // store matches in vector of pairs
    for(Rect outline : outlines)
    {
        Point centerOutline = getRectCenter(outline);
        float minDist = numeric_limits<float>::max();
        Rect rectClosestDesignator;
        for(Rect designator : designators)
        {
            Point centerDesignator = getRectCenter(designator);
            float dist = getPixelDistance(centerOutline, centerDesignator);
            {
                if(dist < minDist)
                {
                    minDist = dist;
                    rectClosestDesignator = designator;
                }
            }
        }
        pairs.emplace_back(pair<Rect, Rect>(outline, rectClosestDesignator));
    }
    return pairs;
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
    Mat imgResistor, imgCapacitor;

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
    openImgFile(imgResistor, pathTplDir + "/resistor.png");
    openImgFile(imgCapacitor, pathTplDir + "/capacitor.png");


    /** Step 1 : template matching for component designators (R, C, D) and R outlines **/
    vector<Rect> matchesR, matchesROutline, matchesC;
    matchesR = findTplMatchesInteractive(imgPcb, imgTplR, 'n', "Resistors");
    matchesROutline = findTplMatchesInteractive(imgPcb, imgTplROutline, 'n', "Resistors Outline");
    matchesC = findTplMatchesInteractive(imgPcb, imgTplC, 'n', "Capacitors");

    /** Step 2 : Use connected component analysis to find the outlines of other components (C, D) **/
    Mat imgGS; // grasycale version of input image
    Mat imgThr, imgThrMorph; // will hold thresholded, and eroded/dilated version of thresholded input image
    vector<Rect> otherOutlines;
    Mat morphKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    namedWindow("Filtered Holes Result");
    int tbOutlineThr = 200;

    // Let user play with threshold values to filter out PCB holes
    // A gray-scale version of the original image is thresholded to select only the silk screen and holes.
    // Then the holes are removed by first eroding away all the thin(compared to the white holes) silk screen lines
    // Alternative: use template matching on holes
    GaussianBlur(imgGS, imgGS, Size(3, 3), 0.0);
    cvtColor(imgPcb, imgGS, COLOR_BGR2GRAY);
    createTrackbar("Outline Threshold Trackbar", "Filtered Holes Result", &tbOutlineThr, 255, nullptr, nullptr);
    while(true)
    {
        threshold(imgGS, imgThr, tbOutlineThr, 255, THRESH_BINARY);
        erode(imgThr, imgThrMorph, morphKernel, Point(-1, -1), 2);
        dilate(imgThrMorph, imgThrMorph, morphKernel, Point(-1, -1), 5);
        imgThr = imgThr & ~imgThrMorph;
        imshow("Filtered Holes Result", imgThr);

        if(waitKey(5) == 'n')
            break;
    }

    // Now let the user play with trackbar to select outlines by area. Only relatively large connected
    // components are outlines, so the user should select a sufficiently large value.
    Mat ccLabels, ccStats, ccCentroids;
    int maxArea, tbCCAreaThr;
    vector<int> ccAreas;

    namedWindow("CC Area Result");
    int numComponents = connectedComponentsWithStats(imgThr, ccLabels, ccStats, ccCentroids);
    for(int i = 1; i < numComponents; i++)
        ccAreas.push_back(ccStats.at<int>(i, CC_STAT_AREA));
    sort(ccAreas.begin(), ccAreas.end());
    for(int area : ccAreas)
        cout << area << endl;
    createTrackbar("CC Area Threshold Trackbar", "CC Area Result", &tbCCAreaThr, ccAreas.back(), nullptr, nullptr);
    while(true)
    {
        Mat ccResult = imgPcb.clone();
        otherOutlines.clear();
        for(int i = 1; i < numComponents; i++)
        {
            Mat tmp = ccLabels.clone();
            if(ccStats.at<int>(i, CC_STAT_AREA) < tbCCAreaThr)
                continue;

            inRange(tmp, i, i, tmp);
            Rect bRect = boundingRect(tmp);
            otherOutlines.emplace_back(bRect);
            rectangle(ccResult, bRect, Scalar(255, 0, 255));
        }
        imshow("CC Area Result", ccResult);
        if(waitKey(5) == 'n')
            break;
    }


    /** match Resistor designators ('R' on the silkscreen) with nearest by Resistor outlines **/
    vector<pair<Rect, Rect> > pairsR = getDesignatorOutlinePairs(matchesROutline, matchesR);

    imgResult = imgPcb.clone();
    for(pair<Rect, Rect> pairR : pairsR)
    {
        Mat resizedResistor;
        resize(imgResistor, resizedResistor, pairR.first.size());
        line(imgResult, getRectCenter(pairR.first), getRectCenter(pairR.second), Scalar(255, 0, 0));
        resizedResistor.copyTo(imgResult(pairR.first));
    }

    /** match Capacitor designators with nearest outline fomr otherOutlines vector **/
    // We match the matched designators with the outlines, as opposed to above, where we match the outlines with the resistors!
    // (Because of this, the items in the pair vector are switched (first <-> second))
    vector<pair<Rect, Rect> > pairsC = getDesignatorOutlinePairs(matchesC, otherOutlines);
    for(pair<Rect, Rect> pairC : pairsC)
    {
        Mat imgDestCapacitor;

        imgDestCapacitor = imgCapacitor.clone();
        if((float) pairC.second.height > pairC.second.width * 1.2)
        {
            rotate(imgCapacitor, imgDestCapacitor, ROTATE_90_CLOCKWISE);
        }
        resize(imgDestCapacitor, imgDestCapacitor, pairC.second.size());
        imgDestCapacitor.copyTo(imgResult(pairC.second));
    }

    imshow("Final Result", imgResult);
    waitKey(0);
}



