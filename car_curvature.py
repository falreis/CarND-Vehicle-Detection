import numpy as np
import cv2

ym_per_pix_curv = 30/220 #720 # meters per pixel in y dimension
xm_per_pix_curv = 3.7/350 #1280 # meters per pixel in x dimension

xm_size_pos = 1280
xm_per_pix_pos = 3.7/1280 #1280 # meters per pixel in x dimension

def curvature(left_fitx, right_fitx, ploty):
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix_curv, left_fitx*xm_per_pix_curv, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix_curv, right_fitx*xm_per_pix_curv, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix_curv + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix_curv + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def relative_position(left_pos, right_pos):
    left = xm_size_pos/2 - left_pos
    right = right_pos - xm_size_pos/2
    
    return (right - left) * xm_per_pix_pos

def write_text(img, curve_rad, position):
    #if radius > 2000m (arbitrary value), we have a straight
    txt_curve = "Curve Radius:" + str('{0:.2f}'.format(curve_rad)) + "m"    
    txt_pos = "Relative Pos: " + str('{0:.2f}'.format(position)) + "m"
    
    draw_img = cv2.putText(img=np.copy(img)
                               , text=(txt_curve + " | " + txt_pos)
                               , org=(10,50)
                               , fontFace=2
                               , fontScale=1.5
                               , color=(255,255,255)
                               , thickness=2
    )
    return draw_img

print('Calculate curvature function OK')