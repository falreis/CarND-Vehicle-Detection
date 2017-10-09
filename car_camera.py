import numpy as np
import cv2

#camera calibration functions
CONST_CORNERS_HORIZ = 9
CONST_CORNERS_VERT = 6

def calibrate_camera(gray_image, objpoints, imgpoints):
    return cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

def undistort(chess_image, gray_image, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(gray_image, objpoints, imgpoints)
    return cv2.undistort(chess_image, mtx, dist, None, mtx)

def chessboard(img, corners=(CONST_CORNERS_HORIZ,CONST_CORNERS_VERT)):
    objpoints = []
    imgpoints = []
    objp = np.zeros((corners[0]*corners[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:corners[0], 0:corners[1]].T.reshape(-1,2)

    #find chessboard corners
    gray = grayscale(img)
    ret, chess_corners = cv2.findChessboardCorners(gray, (corners[0], corners[1]), None)

    #if the chessboard was found
    if ret == True:
        imgpoints.append(chess_corners)
        objpoints.append(objp)
        chess_image = cv2.drawChessboardCorners(img, (corners[0], corners[1]), chess_corners, ret)
    else:
        chess_image = None
    
    return ret, chess_image, objpoints, imgpoints

def calibrate_chess(img):
    corners = [CONST_CORNERS_HORIZ, CONST_CORNERS_VERT]
    ret, chess_image, objpoints, imgpoints = chessboard(img)

    #Different number of corners (try other pattern)!
    if ret == False:
        corners = [9, 5]
        ret, chess_image, objpoints, imgpoints = chessboard(img, corners)
    if ret == False:
        corners = [8, 6]
        ret, chess_image, objpoints, imgpoints = chessboard(img, corners)

    #undistort image
    if ret == True:
        gray_img = grayscale(img)
        undist_img = undistort(chess_image, gray_img, objpoints, imgpoints)
        ret = True
    else:
        ret = False
        undist_img = None
        corners = None
        
    return ret, undist_img, corners

def warped(img, calibrate=True, undist_img=None, corners=None):
    if calibrate == True:
        ret, undist_img, corners = calibrate_chess(img.copy())
    else:
        ret = True
    
    if ret == True:
        nx = corners[0]
        ny = corners[1]
        gray_img = grayscale(img)
        ret, chess_corners = cv2.findChessboardCorners(gray_img, (nx,ny), None)

        if ret == True:
            offset = 100 # offset for dst points
            img_size = (gray_img.shape[1], gray_img.shape[0])
            src = np.float32([chess_corners[0], chess_corners[nx-1], chess_corners[-1], chess_corners[-nx]])
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])
            M = cv2.getPerspectiveTransform(src, dst)
            warp_img = cv2.warpPerspective(undist_img, M, img_size)
        else:
            warp_img = None
    else:
        warp_img = None
        
    return ret, warp_img

print('Camera calibration functions OK')