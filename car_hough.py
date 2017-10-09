import numpy as np
import cv2

#canny, gaussian blur, and hough transform
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def draw_lines(img, lines, color=[255,0,0], thickness=10):
    left_line = []
    right_line = []
    middle_x = img.shape[1] / 2
    
    '''
    Calculate slope of points and create the lines
      - Reject points on the left side of the pictures if the slope is negative
      - Reject points on the right side of the pictures if the slope is positive
    '''
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope > 0: #right
                #reject entry if slope is positive and (x1 or x2) < (middle of image)
                if x1 > middle_x and x2 > middle_x: 
                    right_line.append([x1,y1]) 
                    right_line.append([x2,y2])
            elif slope < 0: #left
                #reject entry if slope is negative and (x1 or x2) > (middle of image)
                if x1 < middle_x and x1 < middle_x:
                    left_line.append([x1,y1]) 
                    left_line.append([x2,y2])
    
    #plot left line
    ldata = np.array(left_line)
    if len(ldata) > 0:
        lfit = np.polyfit(ldata[:,0], ldata[:,1] , 1)
        l1z = np.poly1d(lfit)

        lx1 = CONST_X_LEFT_BOTTOM
        lx2 = CONST_X_LEFT_TOP
        cv2.line(img, (lx1, int(l1z(lx1))), (lx2, int(l1z(lx2))), color, thickness, 4)
    
    #plot right line
    rdata = np.array(right_line)
    if len(rdata) > 0:
        rfit = np.polyfit(rdata[:,0], rdata[:,1] ,1)
        r1z = np.poly1d(rfit)

        rx1 = img.shape[1] - CONST_X_RIGHT_BOTTOM
        rx2 = img.shape[1] - CONST_X_LEFT_TOP
        cv2.line(img, (rx1, int(r1z(rx1))), (rx2, int(r1z(rx2))), color, thickness, 4)

def hough_lines(base_img, canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((canny_img.shape[0], canny_img.shape[1], 3), dtype=np.uint8)
    draw_lines(base_img, lines)
    return base_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.4, β=0.6, λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

#process image (hough)
def process_image(image):
    gray = grayscale(image)
    base_image = np.copy(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold) 

    # This time we are defining a four sided polygon to mask
    vertices = vertices_to_crop(image)
    masked_edges = region_of_interest(edges, vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10 # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 15 # minimum number of pixels making up a line
    max_line_gap = 10 # maximum gap in pixels between connectable line segments    
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    
    hough_image = hough_lines(image, masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
        
    # Draw the lines on the edge image
    w_image = weighted_img(hough_image, base_image)
    
    return w_image

print('Hough function OK')