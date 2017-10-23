## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup/car_notcar1.png
[image2]: ./writeup/car_notcar2.png
[image3]: ./writeup/car_notcar3.png
[image4]: ./test_images/test1.jpg
[image5]: ./test_images/test2.jpg
[image6]: ./test_images/test3.jpg
[image7]: ./test_images/test4.jpg
[image8]: ./test_images/test5.jpg
[image9]: ./test_images/test6.jpg
[image10]: ./writeup/output1.jpg
[image11]: ./writeup/output2.jpg
[image12]: ./writeup/output3.jpg
[image13]: ./writeup/output4.jpg
[image14]: ./writeup/output5.jpg
[image15]: ./writeup/output6.jpg
[image16]: ./writeup/heatmap1.png
[image17]: ./writeup/heatmap2.png
[image18]: ./writeup/heatmap3.png
[image19]: ./writeup/heatmap4.png
[image20]: ./writeup/heatmap5.png
[image21]: ./writeup/heatmap6.png

[image10]: ./examples/labels_map.png
[image11]: ./examples/output_bboxes.png
[video1]: ./output_project_video.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `train_pipeline` of the [IPython notebook](./code.ipynb) and the functions `extract_features` and `hog_extract_features` in the file [udacity_features.py](./udacity_features.py).

I started by reading in all the `vehicle` and `non-vehicle` images, using the function `read_dataset`, in *Helper Functions Section*. I picked only the GTI images, and discard KITTI images, because the results were better using only GTI dataset. Also, I saw some images in KITTI dataset that can be mislead the algorithm, like images from the tires of the car, classified as non-vehicle. Then, I decided to keep only GTI images and here is some examples of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle and Non-vehicle dataset][image1] ![Vehicle and Non-vehicle dataset][image2] ![Vehicle and Non-vehicle dataset][image3]

As the number of vehicles were less than the number of non-vehicles, I evened the number of the vehicles set, replicating some images. The function used to it is `read_dataset`, in the *Helper Functions Section*.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, `cells_per_block`, `hog_channel`, `spatial_size` and `hist_bin`). Here is the final configuration of HOG parameters, using the `YCrCb` color space, that I use to find vehicles:

```python
orient = 7 #11 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 0 #0 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32,32) # Spatial binning dimensions
hist_bins = 32 # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
Exploring some parameters, after some tests in image test set, I realized that the increase of some parameters was causing overtraining. In the example below, the training are better when I set the HOG parameter `cell_per_block=1`. But, when I test it in my image test set, the results were better with `cell_per_block=0` (the worst accuracy in my example below). This happens with many other parameters, but I picked this to ilustrate the situation.

```python
YCrCb Color: (orient: 10, cell_per_block:0) 
1.23 Seconds to train SVC...
Test Accuracy of SVC =  0.9487
-
YCrCb Color: (orient: 10, cell_per_block:1) 
4.42 Seconds to train SVC...
Test Accuracy of SVC =  0.9929
-
YCrCb Color: (orient: 10, cell_per_block:2) 
12.49 Seconds to train SVC...
Test Accuracy of SVC =  0.9891
```

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried several combinations of parameters. I noticed that some parameters were better using `YCrCb` and `YUV` color spaces, combined or separatelly. Other colors spaces like `HSV`, `LUV`, `RGB` and `HLS` color spaces had a lot of false positives then it wasn't a good choice. Using all different color spaces, I tried a lot of differents values for Hog parameters and Hog channels, but the results also generated a lot of false positives.

Using `YUV`, the results were good but I think that I had some difficults to adjust the threshold of the heatmap. `YCrCb` color space resulted in more consistent results than `YUV`. Mixing both of color spaces, the results not improved so much, and the speed of the algorithm decrease due the add of a new color space and extra HOG processment.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only all GTI images, as I said in section 1. The training procedure is available in [IPython notebook](./code.ipynb), in the section *Training Set* and also in the function `train_pipeline` and `read_dataset`, in the functions section.

In *Helper Functions* section, the function `train_pipeline` receive the names of the color spaces that I want to train. This procedures increased the speed of my tests, because I can put some color spaces to run and get the results after a moment, once the training procedure is slow.

One of the main procedure call in the `train_pipeline` function is the procedure `extract_features` (code available in *udacity_features.py*), that reads the dataset and returns the processing sets to be training using the Linear SVC. In this procedure, I used HOG, Spatial and Color Features to compose the features used in the training procedure.

Other approach that I tried, but it hadn't so much success was using a different Support Vector Machine. I tried SVC with RBF kernel but the training process decrease over 10 times and the results were worse than my previous one. I didn't do much tests of RBF Kernel due the poor performance.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search known images over the pictures using different windows sizes. I choosed this approach due the scale transformation over 3D views into 2D images. In 2D images, far away objects is smaller than near objects (perspective). 

I did one loop increasing the window size each step and finding the vehicles in different positions. Firstly, I searched small vehicles over the top of my region of interests, thinking if they were far away, they will be small from the car perspective. In the next loop, I increased the window size and searched again, starting at the top position, but trying to find big vehicles. I did the same procedure with different windows sizes, trying to find nearby vehicles.

The code for this procedure described in the last 2 paragraph is available in `window_pipeline` function, inside [IPython notebook](./code.ipynb) file.

To decide the windows sizes and the overlap parameters, I tested some situations and tuned the parameters. To the *overlap* parameter, I tried different values in my previous tests and after the first test with the my final HOG approach, I just adjust a little bit for a good number, to tune the algorithm.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched known images using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![Output Image 1][image10]
![Output Image 2][image11]
![Output Image 3][image12]
![Output Image 4][image13]
![Output Image 5][image14]
![Output Image 6][image15]

To optimize the performance of the pipeline, I tuned the parameters of the pipeline, to decrease the number of false positives, that can mislead the algorithm. Also, in the video implementation, I kept previous frames (last 10, for example) in memory and use to find the best bounding box. This procedure, developed in the `pipeline` function, can increase stability of the algorithm and help to reduce false positives, working together with the heatmap filter. 

Other approach that I took was jump some frames (each 3, for example), to reduce the processing time and kept the last bounding box found. As the frames vary a little bit from the next one, because video processing uses like 30 frames/sec, processing the all the frames may produce the same result with much more cost.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap in the `heatmap` and `heatimg` functions and set the thresholded to identify vehicle positions, and reduce false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes (`heatimg` function) to cover the area of each blob detected.  

Here are six test images and their corresponding vehicles found and heatmaps:

![Heatmap Image 1][image16]
![Heatmap Image 2][image17]
![Heatmap Image 3][image18]
![Heatmap Image 4][image19]
![Heatmap Image 5][image20]
![Heatmap Image 6][image21]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![Output Image 1][image10]
![Output Image 2][image11]
![Output Image 3][image12]
![Output Image 4][image13]
![Output Image 5][image14]
![Output Image 6][image15]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

##### Problems

One problem that I faced was the difficult to ajust the algorithm. Change HOG parameters increase or decrease the performance so far, and combining with different color spaces caused a lot of tests and experiments. Some parameters were better to find white cars and others better to find dark cars. But in both situations, the results cause many false positives. Then I spend so much time to define my best colorspaces (YCrCb and YUV) and tune the parameters over this color spaces to find the best option.

Other problem that I faced was the time to process the training set. Using some variation of the HOG parameters, the training time increased a lot, even using AWS CPU or GPU based machines. Using SVC's RBF Kernel, the results were so slow that I prefer to continue with Linear SVC.

Other problem that I faced was using the KITTI dataset. As said in section 1, the results were worse and the processing time was slower.

One common mistake of the algorithm that I spent a lot of time to solve was discard false positives. The algorithm, for some reason classified for several times the left lane line (yellow line) as a vehicle. I don't know if the problem is in the training set or the HOG parameters, but happend a lot of time with different parameters. 

To solve the misleading classification, I think that one solution is create a new training set, with more vehicles and try to use different classifications to increase performance.

##### Limitations, Robustness and Future Approachs

One limitation of the algorithm is the difficult to find distant cars. After some distance, the algorithm can't detect the car properly. Also, other difficult of the algorithm is to avoid some false positives. Finnaly, sometimes the algorithm spend some time to identify the car.

To increase the robustness, the algorithm needs to improve far away car detection and solve all false positives, to avoid misleading and cause un unecesssary brake because of the wrong classification.

One future approach is to increase the performance of the algorithm. To process 50sec of video, the algorithm spent almost 3 minutes, then it can be used in real situations. Other implementation should be using a faster language (C++ or Python without Jupyter), and review the code, to remove unecessary step or increase the software performance.

One possible future approach is to use a Neural Network to classify the images, instead of a Linear Support Vector Machine. Other approach is to use a existent framework, like [YOLO](https://pjreddie.com/darknet/yolo/).

