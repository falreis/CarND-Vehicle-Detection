import numpy as np
import cv2

vertices_y_top = 550
vertices_y_bottom = 700

vertices_src = np.float32(
    [[ 200, 680],
     [ 350, 470],
     [ 800, 470],
     [1200, 680]]
)

vertices_dst = np.float32(
    [[ 430, 700],
     [ 0, 0],
     [ 900, 0],
     [ 850, 700]]
)

CONST_VERT_X1 = 400
CONST_VERT_X2 = 850
CONST_VERT_Y1 = 0
CONST_VERT_Y2 = 700

CONST_PIPELINE_S_CHANNEL_THRESH = (100,255)
CONST_REL_POSITION_INDEX = 640
CONST_PIPE_FRAMES_TO_RELOAD = 10

pp_hist = False
pp_ploty = None
pp_left_fit = None
pp_right_fit = None
pp_left_lane = None
pp_right_lane = None
pp_index = 0

pp_last_left_fitx = None
pp_last_right_fitx = None
pp_last_left_lane_inds = None
pp_last_right_lane_inds = None

mtx = []
dist = []

def warp_transform(img, inv=False):
    #compute perspective, transform M
    if inv == False:
        M = cv2.getPerspectiveTransform(vertices_src, vertices_dst)
    else:
        M = cv2.getPerspectiveTransform(vertices_dst, vertices_src)

    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

def warp_pipeline(img):
    global pp_hist, pp_ploty, pp_left_fit, pp_right_fit, pp_left_lane, pp_right_lane, pp_index
    global pp_last_left_fitx, pp_last_right_fitx, pp_last_left_lane_inds, pp_last_right_lane_inds
    
    if pp_index >= CONST_PIPE_FRAMES_TO_RELOAD or pp_hist == False:
        pp_hist = False
        pp_index = 0
    else:
        pp_index += 1
    
    #rename constants to make code more clear
    x1 = CONST_VERT_X1
    x2 = CONST_VERT_X2
    y1 = CONST_VERT_Y1
    y2 = CONST_VERT_Y2
    
    #undistort image
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    #warp original image and s channel image
    warp_orig = warp_transform(undist_img)
    
    #convert to HSL 
    warp_hsl = hls_color(warp_orig)

    #apply sobel and S-Channel conversion
    warp_s2 = binary_s_channel2(warp_hsl, thresh=(90,180))
    warp_left = sobel(warp_s2, sobel_kernel=7, dir_thresh=(0, 2), mag_thresh=(100,200), abs_thresh=(100,200))
    warp_right = sobel(warp_hsl, sobel_kernel=7, dir_thresh=(0, 2), mag_thresh=(50,255), abs_thresh=(50,255))
    warp_conc = np.concatenate((warp_left[:,0:640], warp_right[:, 640:1280]), axis=1)
    
    #define region of interest
    warp_f = np.zeros_like(warp_conc)
    warp_f[y1:y2, x1:x2] = warp_conc[y1:y2, x1:x2]

    #apply fit continuous in the s channel image
    if pp_hist == False:
        out_img = pp_ploty, pp_left_fit, pp_right_fit, pp_left_lane, pp_right_lane = fit_histogram(warp_f)
        pp_hist = True
    
    left_fitx, right_fitx, left_lane_inds, right_lane_inds = fit_continuous(warp_f, True, pp_left_fit, pp_right_fit, pp_left_lane, pp_right_lane)

    #if algorithm didn't find the road, use last one
    min_left_fitx = np.min(left_fitx)
    max_left_fitx = np.min(left_fitx)
    min_right_fitx = np.min(right_fitx)
    max_right_fitx = np.max(right_fitx)
    
    if pp_last_left_fitx != None:
        if (((max_right_fitx - min_left_fitx) > 500) or ((min_right_fitx - max_left_fitx) < 150)):
            left_fitx = pp_last_left_fitx
            right_fitx = pp_last_right_fitx
            left_lane_inds = pp_last_left_lane_inds
            right_lane_inds = pp_last_right_lane_inds
    
    pp_last_left_fitx = left_fitx
    pp_last_right_fitx = right_fitx
    pp_last_left_lane_inds = left_lane_inds
    pp_last_right_lane_inds = right_lane_inds

    #draw green area over the warp image ()
    shape_x = warp_orig.shape[0]
    shape_y = warp_orig.shape[1]
    
    green_warp = warp_orig                
    for i in range(1, warp_orig.shape[0]):
        for j in range(int(left_fitx[i]), int(right_fitx[i])):
            if j<1280 and j>=0:
                green_warp[i,j,0] = 0.
                green_warp[i,j,2] = 0.

    #unwarp image
    green_persp = warp_transform(green_warp,True)

    #merge unwarp image with the original image
    green_persp[0:vertices_y_top, :] = undist_img[0:vertices_y_top, :]
    green_persp[vertices_y_bottom:720, :] = undist_img[vertices_y_bottom:720, :]
    
    #unwarp lane to mesure curvature
    unwarp = warp_transform(green_warp,True)
    unwarp_s2 = binary_s_channel2(unwarp, thresh=(50, 180))
    unwarp_left = sobel(unwarp_s2, sobel_kernel=7, dir_thresh=(0, 2), mag_thresh=(100,200), abs_thresh=(100,200))
    unwarp_right = sobel(unwarp, sobel_kernel=7, dir_thresh=(0, 2), mag_thresh=(50,255), abs_thresh=(50,255))
    unwarp_conc = np.concatenate((unwarp_left[:,0:640], unwarp_right[:, 640:1280]), axis=1)
    
    #write curvature and distance to the center
    plotcurve, left_curvex, right_curvex, left_lane_indsx, right_lane_indsx= fit_lane(unwarp_conc)
    left_curverad, right_curverad = curvature(left_curvex, right_curvex, plotcurve)
    pos = relative_position(left_fitx[CONST_REL_POSITION_INDEX], right_fitx[CONST_REL_POSITION_INDEX])
    
    #print(left_fitx[CONST_REL_POSITION_INDEX], right_fitx[CONST_REL_POSITION_INDEX])
    final_img = write_text(green_persp, left_curverad, pos) #use continuous track line, to increase accuracy
    
    return final_img

print('Warp Pipeline OK')