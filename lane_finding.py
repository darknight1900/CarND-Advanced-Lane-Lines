import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 
import os 

nx = 9
ny = 6
CALIB_IMG_PATH = 'camera_cal/*jpg'
CALIB_OUT_IMG_PATH = 'output_images/chessboard_calib_output/'

# Array to store images used in camera calibration

failed_images = [] # Images we could not find corners 

obj_points  = [] # 3D points in real world 3D space  
img_points  = [] # 2D points in image space 

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def chess_board_image_corners(img_path):
    images = []
    image_names= []
    img_files = glob.glob(img_path)

    objp = np.zeros((nx*ny, 3), dtype= np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for img_file in img_files:
        image = cv2.imread(img_file)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        name  = img_file.split('/')[-1]
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if ret:
            image_names.append(name)
            images.append(image)
            img_points.append(corners)
            obj_points.append(objp)

        else:
            print('Unable to find board corners for file ' + img_file)
            failed_images.append(image)
    return img_points, obj_points

# performs image distortion correction  
def calib_undistort(img_path, out_path, objpoints, imgpoints):
    """
    Undistort image and save the results side by side.
    """
    img_files = glob.glob(img_path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for idx in range(len(images)-1):
        img = images[idx]
        img_size = img.shape[:2][::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        corners = imgpoints[idx]
        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        offset = 100
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, img_size)
        concated = concat_images(img, undist)
        concated = concat_images(concated, warped)
        new_name = image_names[idx].split('.')[0] + '_warped' +'.jpg'
        new_name = out_path + new_name
        print(new_name)
        cv2.imwrite(new_name, concated)

calib_undistort(images, CALIB_OUT_IMG_PATH, obj_points, img_points)
