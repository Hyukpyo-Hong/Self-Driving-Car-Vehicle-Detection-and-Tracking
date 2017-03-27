

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image2-1]: ./output_images/normalized.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG, Color, and Spatial features from the training images.

The code for this step is contained in `second code cell` of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`), `bin_spatial()` parameters (`spatial_size`), and `color_hist()` parameters (`hist_bins`). I grabbed random images from each of the two classes and displayed them to get a feel for what the three functions' output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, Spatial Binning parameters of `spatial_size=(32,32)`, and Color Histogram of 3 channels parameters of `hist_bins=32`.:


![alt text][image2]

####2. Explain how you settled on your final choice of parameters HOG, Color, and Spatial features.

I tried various combinations of parameters and settled on below values, since they are shows 99% of accuracy when evaluate with the test images.
```
color_space = 'YCrCb'
spatial_size = (32, 32)
hist_bins = 32    
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" 
spatial_feat = True
hog_feat = True
hist_feat = True
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I concatenated three features(HOG, Spatial, Color Histogram) into one array. It has 8,460 features of 17,760 images.
Then, I divided this combined features into train set(80%, 14,208 images) and test set(20%, 3,552 images), then trained a linear SVM using `sklearn.svm.LinearSVC(),` and normalized with `sklearn.preprocessing.StandardScaler.transform()` in the first cell of the IPython Notebook. As I mentioned above, the accuracy of test images was 99%.

Here is a histogram of before and after normalized:
![alt text][image2-1]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To searched the features, I slid the windows with three scale factors(1, 1.5, 2). Since small scale factor makes quite small windows, I limited searching region according to the size of scale for processing efficiency. Then by setting parameter `cells_per_step = 2`, each window overlap of 75%. Also, I computed HOG features of the image once then sub-sampled to get all of its overlaying windows so that it improve the processing speed.

To minimize searching area, I also set the x_start,x_end,y_start,and y_end values. I didn't include left side of the image for this project, since there is no car, but if this is for general case, I would include left side. 

Each sub-sampled partial images are processed by the Linear SVM to classify whether it is a car image or not. If it is classified as a car, the region of the window is appended to the whole window list.

All these processes are implemented in `find_cars()` in the first cell of the IPython Notebook.

Here is an image which shows my searching region with different scaled windows. Red: Scale of 1, Green: Scale of 1.5, Blue: Scale of 2.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)



####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections, I created a heatmap and added into the global variable `heatmaps`. Then I calculated mean value of recent 10 heatmaps. then remove the values which are less than 1 by `apply_threshold()`. This reduce number of false positives and make the bounding much smoother.

I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected, and put Text 'Car' on top of the box.

After I figured out proper parameters, I only detect vehicles per every 5 frames to improve processing time.

Here's an example result showing the heatmap from several subsequent images, and the bounding boxes then overlaid on the images:

### Here are sample images and their corresponding heatmaps and final result:

![alt text][image5]


###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The project's test video shows clear whether and the road's color is almost same. But it might be difficult to detect vehicles under bad or dark whether,or on the road of irregular colors. So I will first check which color mode(LUV, HSV, YCrCb, HLS, YUY) works well in hard enviornment. Then, augment my training set to have more various condition such as light, shadow, angle, and so on. 

2. When two vehicles are closed to each other, detecting box regards them as an one vehicle. It will makes worse when there are more than two vehicles. To solove this problem, I think tracking each vehicles movement individually is required, then we can predict their location seperately when they are hidden by other vehicles. 

3. My result couldn't draw vehicle's outline exactly, and that can make difficult to track their direction and location correctly in the future course. I think using machine learning, and other sensor can make more specific results. 

4. Since searching area is not perfectly matched to the image's width, there is the blind area on the right side. That makes difficult to detect the vehicle which runs on blind area continuously. 
A possible solution might be arraying more searching windows from right to left to cover the blind area. 

