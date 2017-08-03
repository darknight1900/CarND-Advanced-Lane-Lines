import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 
import os 

from lane_detect_helper import *

class Line(object):
    """
    Line object
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        # average x values of the fitted line over the last n iterations
        self.bestx = None     
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None


class LaneDetector(object):
    """
    Lane detector object
    """
    def __init__(self, in_video_file, out_video_file):
        self.in_video   = in_video_file
        self.out_video  = out_video_file
        self.left_line  = Line()
        self.right_line = Line()
        self.img_idx = 0;
    def _binary_image(self, image):
        return binarize_image(image)
    def _perspective_transform(self, image):
        return perspective_transform(image)
    def _process_image(self,image):
        return process_image(image)
    
    def process_video(self):
        cap = cv2.VideoCapture(self.in_video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            out_frame = self._process_image(frame)
            cv2.imshow('frame',out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# img_path = 'output_images/test_images_undistored/'
# img_path2 = 'output_images/test_images_undistored/out/'
# img_files = glob.glob(img_path + 'test1.jpg')# +  glob.glob(img_path + '*.png')

ld = LaneDetector('test.mp4', None)
ld.process_video()

# for img_file in img_files:
#     # Load color images into BGR format and convert to RGB 
#     image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

#     result = process_image(image)
#     # image_name = img_file.split('/')[-1].split('.')[0]
#     # image_name = image_name + '_warpped.jpg'
#     # image_name = img_path2 + image_name
#     # warped_image = warped_image*255
#     # warped_image[warped_image > 255] = 255
#     # cv2.imwrite(image_name, warped_image)

#     #Plot the result
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#     f.tight_layout()

#     ax1.imshow(image)
#     name = img_file.split('/')[-1]
#     ax1.set_title(name, fontsize=40)
    
#     ax2.imshow(result)
#     ax2.set_title('result', fontsize=40)
#     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#     plt.show()


