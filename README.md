## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
---

#### 1. Writeup / README

My project includes the following files/folders:
* [code.ipynb](./code.ipynb) containing jupyter notebook file, with the pipeline code of this project
* [udacity_features.py](./udacity_features.py) Contains all features functions like Hog, Spatial and Color features functions
* [udacity_heat.py](./udacity_heat.py) Contain heat and heatmap functions
* [udacity_window.py](./udacity_window.py) Contain slide windows functions
* [/output_images](./output_images/) folder contain all the result image files
* [output_project_video.mp4](./video.mp4) containing the output video file
* [writeup_report.md](./writeup_report.md) summarizing the results

#### 2. Functional code
To view the code and execute it, it's necessary Jupyter Notebook module with CarND-Term1 Anaconda environment. It's possible to see the code in Github too.
