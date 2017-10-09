import numpy as np

#region of interests
CONST_X_LEFT_TOP = 540
CONST_X_LEFT_BOTTOM = 200
CONST_X_RIGHT_BOTTOM = 1200
CONST_Y_TOP = 450
CONST_Y_BOTTOM = 680

def region_of_interest(img, vertices):
    mask = np.zeros_like(img) #defining a blank mask to start with
    
    if len(img.shape) > 2: #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color) #filling pixels inside the polygon with the fill color    
    return cv2.bitwise_and(img, mask)

def vertices_to_crop(img):
    imshape = img.shape
    left_top = (CONST_X_LEFT_TOP,CONST_Y_TOP)
    left_bot = (CONST_X_LEFT_BOTTOM, CONST_Y_BOTTOM)
    right_top = ((imshape[1]-CONST_X_LEFT_TOP), CONST_Y_TOP)
    right_bot = (CONST_X_RIGHT_BOTTOM, CONST_Y_BOTTOM)
    vertices = np.array([[left_bot, left_top, right_top, right_bot]], dtype=np.int32)
    return vertices

print('Region of Interests functions OK')