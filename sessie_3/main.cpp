#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const String keys("{help h    |            |print this message }"
                  "{@input    |recht.jpg   |input image}"
                  "{@template |template.jpg|template image}");

/**
 * Helper functie om input files in te lezen
 */
static void input_open(vector<String> paths, vector<Mat*> pMats)
{
	size_t i;
	if(paths.size() != pMats.size())
	{
		fprintf(stderr, "Error opening input files\n");
		exit(1);
	}
	cerr << paths.size() << endl;
	for(unsigned int i = 0; i < paths.size(); i++)
	{
		cerr << "Opening " << paths[i] << endl;
		*(pMats[i]) = imread(paths[i]);
		if((pMats[i])->empty())
		{
			fprintf(stderr, "Could not open input file %s\n", paths[i].c_str());
			exit(1);
		}

	}

}

// Return the rotation matrices for each rotation
// The angle parameter is expressed in degrees!
void rotate(Mat& src, double angle, Mat& dst)
{
    Point2f pt(src.cols/2., src.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
}

int main(int argc, char * argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img_input, img_template;
    String path_input;
    String path_template;

    if(parser.has("help"))
    {
    	parser.printMessage();
    	return 0;
    }

    path_input = parser.get<String>("@input");
    path_template = parser.get<String>("@template");

    if(!parser.check())
    {
    	parser.printErrors();
    	return 1;
    }

    input_open({path_input, path_template}, {&img_input, &img_template});


    imshow("input", img_input);
    imshow("template", img_template);

    /**
     * Template matching
     **/

    Mat img_tm_result;
    int result_rows = img_input.rows - img_template.rows + 1;
    int result_cols =  img_input.cols - img_template.cols + 1;
    /// result image aanmaken
    img_tm_result.create( result_rows, result_cols, img_input.type() );
    matchTemplate(img_input, img_template, img_tm_result, TM_CCORR_NORMED);
    imshow("match map", img_tm_result);
    /// normalizeren
    /// type = CV_8U voor later gebruik in connectedComponents()
    normalize( img_tm_result, img_tm_result, 255, 0, NORM_MINMAX, CV_8U);



    Mat img_mask(img_input.rows, img_input.cols, CV_8U);

    /*** a) bounding boxes bij alle pixels waar matches > bepaalde threshold ***/
    threshold(img_tm_result, img_mask, 0.96*255, 255, THRESH_BINARY);
    imshow("thresh", img_mask);
    Mat img_result_1 = img_input.clone();
    for(int row = 0; row < img_mask.rows; row++)
    {
        for(int col = 0; col < img_mask.cols; col++)
        {
            if(img_mask.at<uchar>(row, col))
            {
                rectangle(img_result_1, Point(col, row), Point(col + img_template.cols, row + img_template.rows), Scalar(0, 255, 0), 5);
            }
        }
    }
    imshow("Resultaat (1)", img_result_1);

    /** b) bounding box bij globaal maximum */
    Mat img_result_2 = img_input.clone();
    Point minLoc, maxLoc;

    minMaxLoc(img_tm_result, NULL, NULL, &minLoc, &maxLoc);
    rectangle(img_result_2, maxLoc, Point(maxLoc.x + img_template.cols, maxLoc.y + img_template.rows), Scalar(0, 255, 0), 1);
    imshow("Resultaat: 1 match (max)", img_result_2);


    /** c) bounding box bij lokale maxima **/
    Mat img_result_3 = img_input.clone();
    Mat img_labels;
    /// detecteer components (regios > threshold) waarbinnen locale minmax gezocht moeten worden
    int num_components = connectedComponents(img_mask, img_labels, 8);
    for(int i = 0; i < num_components; i++)
    {
        Mat img_mask_component;
        Mat img_search;
        /// maak masker voor de i-de gevonden component
        inRange(img_labels, i, i, img_mask_component);
        /// pas masker toe op match map
        img_search = img_tm_result & img_mask_component;
        /// nu kunnen we minMaxLoc toepassen op de gemaskte afbeelding die enkel
        /// de ene component bevat
        minMaxLoc(img_search, NULL, NULL, NULL, &maxLoc);
        rectangle(img_result_3, maxLoc, Point(maxLoc.x + img_template.cols, maxLoc.y + img_template.rows), Scalar(0, 255, 0), 1);
    }
    imshow("Resultaat: alle matches", img_result_3);

    /** d) Rotated detectie **/
    int max_angle = 90, step_angle = 1;
    int steps = 0;
    vector<Mat> rotated_images;
    vector<Point> matches;

    steps = max_angle / step_angle;
    for(int i = 0; i < steps; i++)
    {
    	Mat rotated = img_input.clone();
    	rotate(img_input, (i+1)*step_angle, rotated);
    	rotated_images.push_back(rotated);
    }

    /** Rotated template matching **/
    Mat img_result_4 = img_input.clone();
    for(int i = 0; i < (int)rotated_images.size(); i++)
    {
    	Mat img_match(result_rows, result_cols, img_input.type());
        Mat img_mask(img_input.rows, img_input.cols, CV_8U);
        Mat img_labels;
        double maxVal = 0.0;

    	matchTemplate(rotated_images[i], img_template, img_match, TM_CCORR_NORMED);
    	normalize(img_match, img_match, 255, 0, NORM_MINMAX, CV_8U);
    	threshold(img_match, img_mask, 254, 255, THRESH_BINARY);
    	int num_components = connectedComponents(img_mask, img_labels, 8);
    	cerr << "Processing image " + to_string(i) << endl;
    	for(int j = 1; j < num_components; j++)
    	{
    		cerr << "   Processing cc " + to_string(j) << endl;
    		Mat img_mask_component;
    		Mat img_search;
    		inRange(img_labels, j, j, img_mask_component);
    		img_search = img_match & img_mask_component;
    		minMaxLoc(img_search, NULL, &maxVal, NULL, &maxLoc);
    		cerr << "       maxVal " << to_string(maxVal) << " at " << maxLoc << endl;
    		rectangle(rotated_images[i], maxLoc, Point(maxLoc.x + img_template.cols, maxLoc.y + img_template.rows), Scalar(0, 255, 0), 1);

    		Point2f pt(img_input.cols/2., img_input.rows/2);
    		Mat r = getRotationMatrix2D(pt, -(step_angle*(i+1)), 1.0);
    		vector<Point2f> pts;
    		pts.push_back(maxLoc);
    		pts.push_back(Point(maxLoc.x + img_template.cols, maxLoc.y));
    		pts.push_back(Point(maxLoc.x + img_template.cols, maxLoc.y + img_template.rows));
    		pts.push_back(Point(maxLoc.x, maxLoc.y + img_template.rows));
    		transform(pts, pts, r);
    		line(img_result_4, pts[0], pts[1], Scalar(255, 0, 0));
    		line(img_result_4, pts[1], pts[2], Scalar(255, 0, 0));
    		line(img_result_4, pts[2], pts[3], Scalar(255, 0, 0));
    		line(img_result_4, pts[4], pts[0], Scalar(255, 0, 0));
    	}
    	//imwrite(to_string(i) + "rotated_match.jpg", rotated_images[i]);
    }

    imshow("result4", img_result_4);

    waitKey(0);
    return 0;
}
