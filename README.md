# Janus
View Dependent Desktop VR using Head Tracking based on Digital Image Processing

Janus was the two-faced god of gates, doors, doorways, beginnings and endings in Roman mythology.

## Abstract

Janus is a program which gives the user an enhanced 3D impression of objects shown on a screen without having to wear special 3D glasses. This is done by adapting the scene shown on the screen according to the angle in which the user looks at the screen. 

Imagine a real-time rendered cube in a 3D environment. When your head is in front of of the screen, the cube is shown from the front. But as you move your head to the right (while still looking at the center of the screen), the cube will begin to turn clockwise, so that you can see its right side. 

You can also imagine your screen as a window through which you are viewing a scene (for example a landscape). When you move left or right you see more of the right/left part of the landscape outside. Moving towards the window shows you more details, while moving away you are seeing a smaller part of the world outside. 

The DIP-part of this is to estimate the viewing angle with help of a webcam mounted above the screen.

## Video

Here's how Janus works now:

[![Janus in Action](http://img.youtube.com/vi/ADhMiJFI3eI/0.jpg)](http://www.youtube.com/watch?v=ADhMiJFI3eI)

And yet another video of the current status, showing the head tracking:

[![headtracking](http://img.youtube.com/vi/rshDUDoSXBg/0.jpg)](http://www.youtube.com/watch?v=rshDUDoSXBg)


## Implementation

The implementation of Janus has three steps (the first two are part of the course project, while the third may be done after the submission):

1. Implementation of the basic structure & testing with a ping-pong ball tracking approach.
2. Implementation of a robust head tracking based on Haar Classifier & its optimization.
3. Render a OpenGL 3D world image.

### Basic structure and ping-pong ball tracking

The first implementation part included the basic structure for the application. We have chosen to use C++ as implementation language and the OpenCV library to support our work. 

The first goal was a quick & dirty implementation which defines the structure for further enhancement, designing everything in modular fashion to enable easy exchangeability of parts of the program and first impression of the wanted project results. 

The main method is responsible for calling the other predefined modules, and creating the help threads (if they are needed). 

Our first approach was based on tracking an orange ping-pong ball attached to the cap of a player. See a demonstration of the idea and it's realization:

[![ping pong ball tracking](http://img.youtube.com/vi/nBXsQ3Fe_Zk/0.jpg)](http://www.youtube.com/watch?v=nBXsQ3Fe_Zk)

For the tracking we convert the image from the RGB space into the HSV space, and then we scan sequentially the HSV image and search for points having a predefined threshold of hue, saturation and intensity (in our case: 100, 120 and 200). Based on the points obtained we are calculating the median of this to be the center of the ping-pong ball. Based on the count of the points we are obtaining the radius of the ball (we are using the function: radius = 6*sqrt(count); in order to have a similar radius as the one we will obtain later on with our head tracking algorithm). The rest is similar to the approach mentioned below. 

Since the identification of the ball took too much time in this first approach which made the picture "jump" from frame to frame making a slightly uncomfortable experience for the user we implemented two POSIX threads using the standard POSIX library of C. The main thread was responsible of drawing the contents of the screen, while the helper thread was doing the work for the ping-pong ball tracking. 

What is more, in order for this to happen we needed to have something to display while the ping-pong ball was being identified. That is why we implemented an interpolation between the current found center point and radius and the ones found in the previous frame. In that way we were doing 5 linear interpolation steps between the two states, making the user think that the face was recognized as fast as the image is drawn on the screen.	

The threads had significant impact on the performance of the program making the solution robust and acceptable.

### Head tracking based on Haar Classifier

Although, the ping-pong ball solution was now good enough it had one major drawback - the user had to wear a cap with an attached orange ping-pong ball, which we found to be looking a little too foolish ;-). 

Therefore, we switched to real head tracking. OpenCV offers a a very good engine for face recognition, based on the Harr classifier. The function cvHaarDetectObjects finds rectangular regions in the given image that are likely to contain objects the cascade has been trained for and returns those regions as a sequence of rectangles. The function scans the image several times at different scales. Each time it considers overlapping regions in the image and applies the classifiers to the regions. It may also apply some heuristics to reduce number of analyzed regions, such as Canny prunning. After it has proceeded and collected the candidate rectangles (regions that passed the classifier cascade), it groups them and returns a sequence of average rectangles for each large enough group. The default parameters (scale_factor=1.1, min_neighbors=3, flags=0) are tuned for accurate yet slow object detection. 

However, we needed faster operations since our program was dealing with a real time stream of images. The proposed settings in the OpenCV documentation did not really optimize the data. So we first scaled the image to be 50% of the original size to improve speed, converted it to grayscale and normalized the histogram to improve the accuracy. Now, with the thread approach mentioned above we could manage to get reasonable results. 

Nevertheless, we continued searching for options to improve the speed and came up with the CV_FIND_BIGGEST_OBJECT flag which could be set in the flag argument. This way the algorithm analyzes only the region which had the biggest found object, which in our case turns out to be exactly the one we want - the face. The parameter however is not available in the current stable release OpenCV 1.0, but in the one available from subversion OpenCV 1.1.0. 

Once we had a nice and robust method for identifying the face we could easily obtain the center of the face, and calculate the radius. Here's an excerpt from the code how:

```cpp
CvPoint center;
int radius, radius_temp;
int radius_old = 0;
for(int i = 0; i < faces->total ; i++) {
    CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
    radius_temp = cvRound((r->width + r->height)*0.25*scale);
    if(radius_temp > radius_old) {
    		center.x = cvRound((r->x + r->width*0.5)*scale);
		   center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = radius_temp;
			radius_old = radius;
    }
}
```

Now we have a radius, and a center and can based on this values render the part of the photograph which represent our perception of the scene.

```cpp
    double speed_shift = .25;
	double speed_zoom = 1.5;
	CvRect slice = cvRect( (speed_shift * (size.width/640) * point->center.x
	                     + (1 - speed_shift) * (size.width/2)) - (size.width/8)
	                     - (speed_zoom * point->radius * ((size.width/4) / 240)),
	                     (speed_shift * (size.height/480) * (480 - point->center.y)
	                      + (1 - speed_shift) * (size.height/2)) - (size.height/8)
	                      - (speed_zoom * point->radius * ((size.height/4) / 240)),
	                      (size.width/4) + (speed_zoom * point->radius
	                      * ((size.width/2) / 240)),
	                      (size.height/4) + (speed_zoom * point->radius
	                      * ((size.height/2) / 240)));                    
	cvSetImageROI(landscape, slice);
```

The `slice` CvSize object is the key to the calculation of the region to be displayed.

There are two parameters here - the two speeds - for the 2D movement (left, right, up, down) we use the `speed_shift` it defines how much the displayed image will be moved based on the movement of the head of the user in each direction.

The second one, the `speed_zoom` likewise specifies the same for the movements towards the screen and away from the screen.

Using the `cvSetImageROI()` function and a subsequent resize to fit the borders of our window we are displaying the correct region. 

However, there was still one problem left and this was the "flackering" of the face recognition algorithm. Although it was identifying the face correct, the radius of the region which was identified was varying very much between two subsequent frames. Here's a small demonstration of the problem:

[![unstable tracking](http://img.youtube.com/vi/c7tjgLMgpts/0.jpg)](http://www.youtube.com/watch?v=c7tjgLMgpts)

As you can see, although the "flackering" seems fine if we are viewing the captured frame and the drawed circle, it has major impact on the real image. Terefore, we needed a solution to this. We discussed several different strategies, such as further optimizing the algorithm by parameters, building our own, or enhancing and optimizing the input frame. All of these would however use too much computational power and computational time. That's why we tried to find a solution based on the data we already have and came up with the following idea: instead of observing only the last captured frame store the data from the last few frames and compute the average. For this we introduced a vector which is storing the last 7 frames and is modified in the identifyHead function.


```cpp
    vector<struct Coords> temp_vector = *pheadvector;
	temp_vector.insert(temp_vector.begin(),temp);
	if (temp_vector.size() > STABILITY) {
		temp_vector.erase(temp_vector.end());
	}
	*pheadvector = temp_vector;
```	

The drawing function then consideres all elements currently in the vector, calculates the average and shows it. The result was wonderful:

[![Janus in Action](http://img.youtube.com/vi/ADhMiJFI3eI/0.jpg)](http://www.youtube.com/watch?v=ADhMiJFI3eI)

## Future work: Render an OpenGL 3D world image

As seen above we have managed to achieved the desired goal for a 2D image, without the need of having too heavy calculations. In fact the optimization of the algoritms allowed us to drop the threads approach. However, life is not that simple in the 3D world. For an OpenGL scene we need to calculate a rotation and a translation matrix, from which we can derive a position matrix. This will not be able to happen using only the approach from above. Here are some of our ideas for the future implementation:

First, after having identified the face and it's radius let us identify in which direction it is currently looking (we were assuming till now that the eyes are directed to the center of the screen. This however is not always the case. When you move left you also turn your head to the right and thus you are now longer looking exactly at the center of the screen, but rather to a point on the right hand side). To do this we first need to have a few popular points. This can be obtained using the `cvFindGoodFeaturesToTrack()` function of OpenCV. Here's how it looks like:

[![Enhanced Headtracking](http://img.youtube.com/vi/sYR2zPuuTmE/0.jpg)](http://www.youtube.com/watch?v=sYR2zPuuTmE)

The next step would be to use this Good Features to Track obtained points as an argument to the `cvCalcOpticalFlowPyrLK()` function. The result points obtained can be used as an input for (the POSIT algorithm for) the calculation of the translation matrix and rotation vector. The last step would be to give them to the GL rendering engine.

## Obtaining the source code, compilation & running of the program

As a prerequisite you will need to have on your system OpenCV of a version 1.1 or above. Otherwise compilation will fail. To run the program you will also need to have a camera attached to the system and identified by your system as a camera source. If this criteria is not met, the program will terminate directly after running informing you about the problem. 

If you already obtained the source code you will see that a Makefile is included. You just have to type make and an executable with the name janus will be complied. Running it will start the program. 

Beware. Termination of the program is currently not implemented properly. You can terminate it by killing the process using the terminal or by issuing the kill command.

## Ideas, creative support, code contribution

In case you want to contact us with some bright ideas, or simply want to work with us, or just criticise or praise us please open an issue.

The project is open source and we would be really happy to hear from you if you use our source code in your application. Do not hesitate to contact us and inform us about that.

## Thank you

If you are reading this you must be our instructor, or teaching assistant, or really love our work. Either way, thank you for giving us the opportunity to present you our ideas and work and we hope to hear from you soon.

## References

1. D.F. DeMenthon and L.S. Davis. Model-Based Object Pose in 25 Lines of Code. International Journal of Computer Vision, Vol. 15, pp. 123-141, June 1995. 
2. OpenCV. Documentation. 2008. 
3. Paul Viola and Michael J. Jones. Rapid Object Detection using a Boosted Cascade of Simple Features. IEEE CVPR, 2001. 
4. Rainer Lienhart and Jochen Maydt. An Extended Set of Haar-like Features for Rapid Object Detection. IEEE ICIP 2002, Vol. 1, pp. 900-903, Sep. 2002.


