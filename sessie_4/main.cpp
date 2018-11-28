#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace cv::detail;

const String keys("{help h usage ? | | print this message }"
                  "{@input         |<none>| input file}"
                  "{@template      |<none>| template file}");

int main(int argc, char * argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img_input_scene, img_input_template;

    String path_input = parser.get<String>("@input");
    String path_template = parser.get<String>("@template");

    if(path_input.empty() or path_template.empty())
    {
        cerr << "Please provide all arguments" << endl;
        return 1;
    }

    img_input_scene = imread(path_input);
    if(img_input_scene.empty())
    {
        cerr <<  "Could not open or find the input image with path '" + path_input + "'" << std::endl ;
        return -1;
    }
    img_input_template = imread(path_template);
    if(img_input_template.empty())
    {
        cerr <<  "Could not open or find the template image with path '" + path_input + "'" << std::endl ;
        return -1;
    }

    //imshow("input", img_input_scene);
    //imshow("template", img_input_template);

    Mat img_keypoints_scene_orb = img_input_scene.clone();
    Mat img_keypoints_scene_brisk = img_input_scene.clone();
    Mat img_keypoints_scene_akaze = img_input_scene.clone();
    Mat img_keypoints_template_orb = img_input_template.clone();
    Mat img_keypoints_template_brisk = img_input_template.clone();
	Mat img_keypoints_template_akaze = img_input_template.clone();

    vector<Mat> imgs;
    imgs.push_back(img_input_scene);
    imgs.push_back(img_input_template);

    /** ORB **/
    vector<KeyPoint> keypoints_orb_template, keypoints_orb_scene;
    Mat descriptors_orb_template, descriptors_orb_scene;

    /// keypoints en descriptors berekenen
    Ptr<ORB> orb = ORB::create();
    orb->detectAndCompute(img_input_scene, Mat(), keypoints_orb_scene, descriptors_orb_scene);
    orb->detectAndCompute(img_input_template, Mat(), keypoints_orb_template, descriptors_orb_template);
    drawKeypoints(img_input_scene, keypoints_orb_scene, img_keypoints_scene_orb);
    drawKeypoints(img_input_template, keypoints_orb_template, img_keypoints_template_orb);
    //imshow("ORB keypoints", img_keypoints_orb);


    /** BRISK **/
    vector<KeyPoint> keypoints_brisk_template, keypoints_brisk_scene;
    Mat descriptors_brisk_template, descriptors_brisk_scene;

    /// keypoints en descriptors berekenen
    Ptr<BRISK> brisk = BRISK::create();
    brisk->detectAndCompute(img_input_scene, Mat(), keypoints_brisk_scene, descriptors_brisk_scene);
    brisk->detectAndCompute(img_input_template, Mat(), keypoints_brisk_template, descriptors_brisk_template);
    drawKeypoints(img_input_scene, keypoints_brisk_scene, img_keypoints_scene_brisk);
    drawKeypoints(img_input_template, keypoints_brisk_template, img_keypoints_template_brisk);
    //imshow("BRISK keypoints", img_keypoints_brisk);

    /** AKAZE **/
    vector<KeyPoint> keypoints_akaze_template, keypoints_akaze_scene;
    Mat descriptors_akaze_template, descriptors_akaze_scene;

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img_input_scene, Mat(), keypoints_akaze_scene, descriptors_akaze_scene);
    akaze->detectAndCompute(img_input_template, Mat(), keypoints_akaze_template, descriptors_akaze_template);
    drawKeypoints(img_input_scene, keypoints_akaze_scene, img_keypoints_scene_akaze);
    drawKeypoints(img_input_template, keypoints_akaze_template, img_keypoints_template_akaze);

    Mat img_features_scene, img_features_template;
    hconcat(img_keypoints_scene_orb, img_keypoints_scene_brisk, img_features_scene);
    hconcat(img_features_scene, img_keypoints_scene_akaze, img_features_scene);
    hconcat(img_keypoints_template_orb, img_keypoints_template_brisk, img_features_template);
    hconcat(img_features_template, img_keypoints_template_akaze, img_features_template);
    imshow("Keypoints in scene: ORB - BRISK - AKAZE", img_features_scene);
    imshow("Keypoints in template: ORB - BRISK - AKAZE", img_features_template);

    //imshow("AKAZE keypoints", img_keypoints_akaze);


    /// Verder: keypoints matchen: probleem: meerder objecten, opl: eerst brute force matching met treshold, dan RANSAC ipv eerst RANSAC

    /// RANSAC: findHomography()
    /// bounding box tekenen rond match: hoekpunten template transformeren volgens homography die je met RANSAC hebt gevonden en dan
    /// lijnen tussen getransformeerde punten tekenen

    /** Matches berekenen via bruteforce (enkel voor ORB gedaan hier, maar gelijkaardig voor andere feature algoritmes) **/
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
	vector<DMatch> matches;
	matcher->match(descriptors_orb_template, descriptors_orb_scene, matches);

    double max_dist = 0; double min_dist = 1000.0;
    Mat img_matches;
    do {
		/// Minimum en maximum afstand tussen matches berekenen
		for( int i = 0; i < matches.size(); i++ )
		{
			double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}

		fprintf(stderr, "Min distance: %f\n", min_dist);
		fprintf(stderr, "Max distance: %f\n", max_dist);
		/// enkel goede matches selecteren (dist <= threshold)
		vector<DMatch> good_matches;
		float dist_threshold = 3*min_dist;
		fprintf(stderr, "Started with %d matches\n", matches.size());
		for(int i = 0; i < matches.size(); i++)
		{
			if(matches[i].distance <= dist_threshold)
			{
				good_matches.push_back(matches[i]);
				// verwijder deze match ook uit de matches vector voor de volgende iteratie van de do-while loop
				matches.erase(matches.begin() + i);
				i++;
			}
		}
		fprintf(stderr, "%d matches remaining\n", matches.size());

		/// matches tekenen
		drawMatches( img_input_template, keypoints_orb_template, img_input_scene, keypoints_orb_scene,
					 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		imshow("ORB brute force matches", img_matches);




		/** Object detectie mbv RANSAC/homografie **/
		std::vector<Point2f> tpl;
		std::vector<Point2f> scene;
		/// keypoint coordinaten van de good_matches ophalen
		for( size_t i = 0; i < good_matches.size(); i++ )
		{
			tpl.push_back( keypoints_orb_template[ good_matches[i].queryIdx ].pt );
			scene.push_back( keypoints_orb_scene[ good_matches[i].trainIdx ].pt );
		}
		Mat H = findHomography( tpl, scene, RANSAC );
		/// Coordinaten van hoekpunten template
		std::vector<Point2f> tpl_corners(4);
		tpl_corners[0] = cvPoint(0,0);
		tpl_corners[1] = cvPoint(img_input_template.cols, 0);
		tpl_corners[2] = cvPoint(img_input_template.cols, img_input_template.rows);
		tpl_corners[3] = cvPoint(0, img_input_template.rows);
		std::vector<Point2f> scene_corners(4);
		/// transformeer coordinaten van hoekpunten template volgens de gevonden homografie
		perspectiveTransform( tpl_corners, scene_corners, H);
		/// Tekenen lijnen tussen getransformeerde hoekpunten
		line( img_matches, scene_corners[0] + Point2f( img_input_template.cols, 0), scene_corners[1] + Point2f( img_input_template.cols, 0), Scalar(0, 255, 0), 4 );
		line( img_matches, scene_corners[1] + Point2f( img_input_template.cols, 0), scene_corners[2] + Point2f( img_input_template.cols, 0), Scalar( 0, 255, 0), 4 );
		line( img_matches, scene_corners[2] + Point2f( img_input_template.cols, 0), scene_corners[3] + Point2f( img_input_template.cols, 0), Scalar( 0, 255, 0), 4 );
		line( img_matches, scene_corners[3] + Point2f( img_input_template.cols, 0), scene_corners[0] + Point2f( img_input_template.cols, 0), Scalar( 0, 255, 0), 4 );
    }
	while (min_dist < 100.0);

    fprintf(stderr, "%f\n", min_dist);



    imshow( "Object detection", img_matches );

    /** Bovenstaande detecteert maar 1 object (beste)
     *  Om meerder objecten te detecteren, idee:
     *  	Keypoints van eerste detectie wegsmijten en opnieuw good matches detecteren
     *  	repeat tot the min distance onder threshold zitten
     */



    waitKey(0);
    return 0;
}
