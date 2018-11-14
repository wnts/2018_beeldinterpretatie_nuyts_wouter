#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>

using namespace std;
using namespace cv;

const String keys("{help h usage ? | | print this message }"
                  "{@image         | | image file}");
int th_val_h_upper = 160, th_val_h_lower = 10, th_val_s = 240;
int update_cc = 0;

static void on_trackbar_h_lower(int val, void * ptr)
{
    th_val_h_lower = val;
    printf("Masking H between 0-%d and %d-255\n", th_val_h_lower, th_val_h_upper);
    update_cc = 1;
}

static void on_trackbar_h_upper(int val, void * ptr)
{
    th_val_h_upper = val;
    update_cc = 1;
    printf("Masking H between 0-%d and %d-255\n", th_val_h_lower, th_val_h_upper);
}

static void on_trackbar_s(int val, void * ptr)
{
    th_val_s = val;
    update_cc = 1;
    printf("Masking S between %d-255\n", th_val_s);
}

int main(int argc, char * argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img;
    char *path = NULL;

    path = getcwd(NULL, 0);
    printf("cwd: %s\n", path);
    String imgpath = parser.get<String>("@image");

    img = imread(imgpath);
    if(img.empty())
    {
        cout <<  "Could not open or find the image with path '" + imgpath + "'" << std::endl ;
        return -1;
    }
    //resize(img, img, Size(img.cols/2, img.rows/2));
    imshow("Oorspronkelijke afbeelding", img);




    /**
     * BGR Segmentatie
     * Nadeel: om een bepaalde kleur te segmenteren in BGR moeten we beperkingen leggen op alle 3 dimensies in de kleurenruimte (cfr. BGR kubus)
     **/

    /*
    vector<Mat> channels_bgr;
    split(img, channels_bgr);

    imshow("Rode component", channels_bgr[2]);
    Mat img_r_thres;
    int th_val_r = 196;
    threshold(channels_bgr[2], img_r_thres, th_val_r, 255, THRESH_BINARY);
    imshow("Thresholded (masker)", img_r_thres);


    Mat result_blauw = channels_bgr[0] & img_r_thres;
    Mat result_green = channels_bgr[1] & img_r_thres;
    Mat result_red   = channels_bgr[2] & img_r_thres;
    Mat result_channels_bgr[] = {result_blauw, result_green, result_red};
    int fromTo[] = {0,0, 1,1, 2,2};
    Mat result_bgr(img.rows, img.cols, CV_8UC3);
    mixChannels(result_channels_bgr, 3, &result_bgr, 1, fromTo, 3);
    imshow("Resultaat (RGB segmentatie)", result_bgr);
*/
    /**
     * Segmetatie in HSV kleurruimte
     * Voordeel: de kleur is vervat in 1 dimensie (Hue), en we moeten dus enkel een beperking op die dimensie
     * leggen om een bepaalde kleur te segmenteren.
     * We leggen echter ook een beperking op de Saturation dimensie omdat de rode kleur in een verkeersbord nogal gesatureerd is.
     **/
    /* input afbeelding omzetten BGR->HSV */
    Mat img_hsv;
    try {
        cvtColor(img, img_hsv, COLOR_BGR2HSV);
    }
    catch(const std::exception& e)
    {
        printf("ERROR: %s\n", e.what());
    }
    vector<Mat> channels_hsv;
    split(img_hsv, channels_hsv);

    Mat h_mask, h_mask_1, h_mask_2, s_mask, result_mask;
    /* Trackbars voor gebruikerinput. De gebruiker kan het interval voor de Hue dimensie kiezen, alsook
       de minimum waarde voor de saturatie */
    namedWindow("H thresh", WINDOW_AUTOSIZE);
    namedWindow("S thresh", WINDOW_AUTOSIZE);
    createTrackbar("Upper H thresh", "H thresh", &th_val_h_upper, 180, on_trackbar_h_upper, NULL);
    createTrackbar("Lower H thresh", "H thresh", &th_val_h_lower, 180, on_trackbar_h_lower, NULL);
    createTrackbar("S thresh", "S thresh", &th_val_s, 255, on_trackbar_s, NULL);
    while(true)
    {
        // druk q om af te sluiten
        if(waitKey(2) == 'q')
            return 0;
        // Segmenteer pixels obv Hue
        inRange(channels_hsv[0], 0, th_val_h_lower, h_mask_1);
        inRange(channels_hsv[0], th_val_h_upper, 180, h_mask_2);
        // segmenteer pixels obv Saturation
        inRange(channels_hsv[1], th_val_s, 255, s_mask);
        h_mask = h_mask_1 | h_mask_2;
        imshow("H thresh", h_mask);
        imshow("S thresh", s_mask);
        // Maak combinatie masker van hue en saturation segmentatie
        result_mask = h_mask & s_mask;
        imshow("H+S masker", result_mask);

        // Pas dilatie-erosie toe om kleine "rommel" op te kuisen
        Mat diler_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(result_mask, result_mask, diler_kernel, Point(-1, -1), 10);
        erode(result_mask, result_mask, diler_kernel, Point(-1, -1), 10);
        imshow("H+S masker na dilatie-erosie", result_mask);


        /*** Connected components analyse met connectedComponents() ***/
        /*
        if(update_cc)
        {
            update_cc = 0;
            Mat ccLabels;
            int nLabels = 0;

            nLabels = connectedComponents(result_mask, ccLabels, 8);
            std::vector<Vec3b> cc_colors(nLabels);
            cc_colors[0] = Vec3b(0, 0, 0);//background
            for(int label = 1; label < nLabels; ++label)
                cc_colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );

            Mat cc_result(result_mask.size(), CV_8UC3);
            for(int r = 0; r < cc_result.rows; ++r){
                for(int c = 0; c < cc_result.cols; ++c){
                    int label = ccLabels.at<int>(r, c);
                    Vec3b &pixel = cc_result.at<Vec3b>(r, c);
                    pixel = cc_colors[label];
                 }
             }

             imshow("Connected components result", cc_result);
        }
        */



        /*** Connected component analyse met findContours() ***/
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours(result_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        // loop over alle contours en vindt grootste
        // We gaan ervan uit dat dit het verkeersbord is
        vector<Point> contour_grootste;
        contour_grootste = contours[0];
        for(int i = 0; i < contours.size(); i++)
        {
            if(contourArea(contours[i]) > contourArea(contour_grootste))
                contour_grootste = contours[i];
        }
        // convexHull rond grootste contour, dan tekenen met drawContours (stop eerst opnieuw in vector van vector van points)
        vector<Point> hull;
        convexHull(contour_grootste, hull);

        vector< vector<Point> > tmp;

        tmp.push_back(hull);
        Mat result = img.clone();
        drawContours(result, tmp, -1, Scalar(0, 255, 0), 3);
        imshow("Resultaat", result);

    }

    return 0;
}
