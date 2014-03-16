/*  This file is part of Janus.

    Janus is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    Janus is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
// #include <pthreds.h>

const char*    CASCADE_NAME = "haarcascade_frontalface_alt.xml"; 
const char*    landscape_NAME = "landscape.jpg";
const bool TRACKINGPOINTS = false;
const bool BALL = false;
const int STABILITY = 7;

using namespace std;

CvCapture * camera;
struct Coords {CvPoint center; int radius;};
struct Coords_Pile {struct Coords coords; struct Coords* next;};
struct IdentifyHeadArgs {IplImage* img; CvMemStorage* storage; CvHaarClassifierCascade* cascade; struct Coords *phead;};
pthread_t identifyHeadThread;

void initializeCamera() {
	camera = cvCreateCameraCapture (CV_CAP_ANY);
	if (! camera) {
        printf("Camera not found!\n");
		exit(1);
	}
}

void identifyBall(IplImage* current_frame, vector<struct Coords>* pheadvector) {
	CvSize frame_size = cvGetSize(current_frame);
	CvScalar c[3];
	int count = 1;
	int stable = 0;
	CvPoint median = cvPoint(0,0);
	CvPoint median_old = cvPoint(0,0);
	
	IplImage* current_frame_hsv = cvCreateImage(frame_size,IPL_DEPTH_8U,3);
	cvCvtColor(current_frame, current_frame_hsv, CV_RGB2HSV);
	
	c[0] = cvGet2D(current_frame_hsv, 1, 1);
	c[1] = cvGet2D(current_frame_hsv, 0, 1);
	for (int j = 1; j < frame_size.height - 2; j=j+2) {
		for (int i = 1; i < frame_size.width - 2; i=i+2) {
			c[1] = c[0];
			c[0] = c[2];
			c[2] = cvGet2D(current_frame_hsv, j+1, i);

			stable = (int)(c[0].val[0] > 100 && c[0].val[0] < 120 && c[0].val[1] > 200)
				+ (int)(c[1].val[0] > 100 && c[1].val[0] < 120 && c[1].val[1] > 200)
				+ (int)(c[2].val[0] > 100 && c[2].val[0] < 120 && c[2].val[1] > 200);
			if (stable > 2	) {
				c[0].val[0] = 240;
				c[0].val[1] = 100;
				c[0].val[2] = 100;
				// cvSet2D(current_frame_hsv, j, i, c[0]);
				count++;
				median.x = median.x + i;
				median.y = median.y + j;
				
			}
		}
	}
	
	median.x = median.x / count;
	median.y = median.y / count;

	struct Coords temp;
	temp.center.x = median.x;
	temp.center.y = median.y;
	temp.radius = 6*sqrt(count);

	vector<struct Coords> temp_vector = *pheadvector;
	temp_vector.insert(temp_vector.begin(),temp);
	if (temp_vector.size() > STABILITY) {
		temp_vector.erase(temp_vector.end());
	}
	
	*pheadvector = temp_vector;
	
	// cvCvtColor(current_frame_hsv, current_frame, CV_HSV2RGB);
	
	cvReleaseImage(&current_frame_hsv);
		
} 

void identifyHead(IplImage* img, CvMemStorage* storage, CvHaarClassifierCascade* cascade, vector<struct Coords>* pheadvector) {
	double scale = 3;
    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
    IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),
                         cvRound (img->height/scale)), 8, 1 );
	cvCvtColor( img, gray, CV_BGR2GRAY );
    cvResize( gray, small_img, CV_INTER_LINEAR );
    cvEqualizeHist( small_img, small_img );

	cvClearMemStorage( storage );

	CvSeq* faces = cvHaarDetectObjects(small_img, cascade, storage,
											1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_DO_CANNY_PRUNING,
                                            cvSize(30, 30) );
	
	if (faces->total == 0)
		return;
		
	CvPoint center;
	int radius, radius_temp;
	int radius_old = 0;
	for( int i = 0; i < faces->total ; i++ ) {
		CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
		radius_temp = cvRound((r->width + r->height)*0.25*scale);
		if(radius_temp > radius_old) {
            		center.x = cvRound((r->x + r->width*0.5)*scale);
		        center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = radius_temp;
			radius_old = radius;
		}
	}
	
	
	struct Coords temp;
	temp.center = center;
	temp.radius = radius;
	vector<struct Coords> temp_vector = *pheadvector;
	temp_vector.insert(temp_vector.begin(),temp);
	if (temp_vector.size() > STABILITY) {
		temp_vector.erase(temp_vector.end());
	}
	
	*pheadvector = temp_vector;
	
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);

}

/*
void* identifyHeadThreadFunction (void* pargs) {
	struct IdentifyHeadArgs* pargs2;
	pargs2 = (struct IdentifyHeadArgs*)pargs;
	identifyHead(pargs2->img, pargs2->storage, pargs2->cascade, pargs2->phead);
	pthread_exit(NULL);
}
*/

int findTrackingPoints(IplImage* current_frame, struct Coords point, CvPoint2D32f* points) {
	CvRect slice = cvRect(point.center.x - (point.radius), point.center.y - (point.radius), 2 * point.radius, 2 * point.radius);
	IplImage* gray = cvCreateImage( cvSize(2 * point.radius, 2 * point.radius), current_frame->depth, 1 );

	cvSetImageROI(current_frame, slice);	
	cvCvtColor(current_frame, gray, CV_RGB2GRAY);
	cvResetImageROI(current_frame);
	
	IplImage* eig = cvCreateImage( cvGetSize(gray), 32, 1 );
	IplImage* temp = cvCreateImage( cvGetSize(gray), 32, 1 );
	
	double quality = .06;
	double min_distance = 5;

	int numPoints = 200;
	
	cvGoodFeaturesToTrack( gray, eig, temp, points, &numPoints, quality, min_distance, 0, 3, 0, 0.04 );	
	
	for (int k = 0; k < numPoints; k++) {
		points[k].x += point.center.x - point.radius;
		points[k].y += point.center.y - point.radius;
	}
		
	
	return numPoints;
}

void drawContent(struct Coords* point, CvPoint2D32f* trackingpoints, int numPoints, IplImage* landscape, IplImage* current_frame) {
	double speed_shift = .25;
	double speed_zoom = 1.5;
	CvSize size = cvGetSize(landscape);
	IplImage* slice_landscape = cvCreateImage( cvSize(800,600),IPL_DEPTH_8U,3);
	
	CvRect slice = cvRect( (speed_shift * (size.width/640) * point->center.x + (1 - speed_shift) * (size.width/2)) - (size.width/8) - (speed_zoom * point->radius * ((size.width/4) / 240)),
							(speed_shift * (size.height/480) * (480 - point->center.y) + (1 - speed_shift) * (size.height/2)) - (size.height/8) - (speed_zoom * point->radius * ((size.height/4) / 240)),
							(size.width/4) + (speed_zoom * point->radius * ((size.width/2) / 240)),
							(size.height/4) + (speed_zoom * point->radius * ((size.height/2) / 240)));
	
	cvSetImageROI(landscape, slice);
	cvResize( landscape, slice_landscape, CV_INTER_LINEAR );
	
	cvCircle( current_frame, point->center, point->radius, CV_RGB( 255, 0, 0 ), 3, 8, 0 );
	
	if (TRACKINGPOINTS) {
		for(int k = 0; k < numPoints; k++)
			cvCircle( current_frame, cvPointFrom32f(trackingpoints[k]), 5, CV_RGB( 0, 0, 255 ), 3, 8, 0);
	}
	
	if (!TRACKINGPOINTS) {
		IplImage* current_frame_small = cvCreateImage( cvSize(320,240),IPL_DEPTH_8U,3);
		cvResize(current_frame, current_frame_small, CV_INTER_LINEAR);
		cvShowImage( "current_frame", current_frame_small );
	}
	else
		cvShowImage("current_frame", current_frame);
	
	cvShowImage( "landscape", slice_landscape );
	cvResetImageROI(landscape);
	cvReleaseImage(&slice_landscape);
	
	cvWaitKey (1);

}

void interpolateAndDraw(vector<struct Coords>* pheadvector, IplImage* landscape, IplImage* current_frame) {
	vector<struct Coords> headvector = * pheadvector;
	struct Coords point = {cvPoint(0,0),0};
	
/*	printf("quotient: %f, head.radius: %i, head_old.radius: %i \n", (double)head.radius / (double)head_old.radius, head.radius, head_old.radius);
	if ((double)head.radius / (double)head_old.radius > 1.5 || (double)head_old.radius / (double)head.radius > 1.5) {
		head.radius = head_old.radius;
	}
*/	
	for(int k = 0; k < headvector.size(); k++) {
		struct Coords elem = headvector.at(k);
		point.center.y += elem.center.y;
		point.center.x += elem.center.x;
		point.radius += elem.radius;	
	}
	
	point.center.y = point.center.y/headvector.size();
	point.center.x = point.center.x/headvector.size();
	point.radius = point.radius/headvector.size();
	
	CvPoint2D32f* trackingpoints = (CvPoint2D32f*)cvAlloc(200*sizeof(trackingpoints[0]));
	int numPoints = 0;
	if (TRACKINGPOINTS)
		numPoints = findTrackingPoints(current_frame, point, trackingpoints);
	
	drawContent(&point, trackingpoints, numPoints, landscape, current_frame);
	
	
}

int main (int argc, char * const argv[]) {	
	
    	cvNamedWindow ("landscape", CV_WINDOW_AUTOSIZE);
	cvNamedWindow ("current_frame", CV_WINDOW_AUTOSIZE);
	
	initializeCamera();
	IplImage* current_frame;
	IplImage* landscape = cvLoadImage (landscape_NAME,1);
	IplImage* window_content = cvCreateImage(cvSize(640,400),IPL_DEPTH_8U,3);
	CvMemStorage* storage = cvCreateMemStorage(0);
    	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*) cvLoad (CASCADE_NAME, 0, 0, 0);

	
	vector<struct Coords> headvector;
	vector<struct Coords>* pheadvector;
	pheadvector = &headvector;
	
	struct Coords init ={cvPoint(0,0),1};
	headvector.insert(headvector.begin(),init);

	// as long as there are images ...
    	while (current_frame = cvQueryFrame (camera)) {
		if(BALL)
			identifyBall(current_frame, pheadvector);
		else
			identifyHead(current_frame, storage, cascade, pheadvector);
		
		interpolateAndDraw(pheadvector, landscape, current_frame);
		
/*		pthread_create (&identifyHeadThread, NULL, identifyHeadThreadFunction, (void *)pargs);
		
		interpolateAndDraw(phead, phead_old, landscape);
		
		pthread_join (identifyHeadThread, NULL);
*/ // Threads

		
		
		int key = cvWaitKey (5);
		if (key == 'q' || key == 'Q')
			break;
		
	}
	
	cvReleaseImage(&landscape);

	return 0;
	
}
