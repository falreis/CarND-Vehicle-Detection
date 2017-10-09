import numpy as np
import cv2

#color functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def hls_color(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def binary_s_channel(img, thresh=(40,255)):
    S = img[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary

def binary_s_channel2(img, thresh=(40,255)):    
    S = img[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 255

    value = np.zeros_like(img)
    value[:,:,2] = binary

    return value

def sobel(img, sobel_kernel=3, dir_thresh=(0, np.pi/2), mag_thresh=(0, 255), abs_thresh=(0,255)):
    #rgb = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    gray = grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    #abs sobel
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    scaledx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaledy = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    #direction sobel
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    arctan = np.arctan2(abs_sobely, abs_sobelx)
    
    #gradmag sobel
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    #combined direction sobel with gradmag sobel
    sxbinary = np.zeros_like(arctan)
    sxbinary[(arctan >= dir_thresh[0]) & (arctan <= dir_thresh[1]) & 
             (gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1]) &
             (scaledx >= abs_thresh[0]) & (scaledx <= abs_thresh[1])
            ] = 1
    # (scaledy >= abs_thresh[0]) & (scaledy <= abs_thresh[1])
    
    return sxbinary

print('Color functions OK')