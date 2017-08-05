import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 
import os 
from lane_detect_helper import *
from moviepy.editor import ImageSequenceClip

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
        # polynomial coefficients for last 5 iterations 
        self.history_fits = np.zeros((5,3))    
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

    # calculate and return best fit
    def calculate_best_fit(self):
        sum_fits = 0
        n_fits = 0
        for i in range(1, self.history_fits.shape[0]):
            if np.count_nonzero(self.history_fits[i]) == 0:
                break
            n_fits += 1
            sum_fits += self.history_fits[i]

        # print(sum_fits, n_fits)
        if n_fits == 0:
            self.best_fit = self.current_fit
        else:
            avg_prev_fits = sum_fits / n_fits
            # low pass filter the fits
            # put more weights on current fit
            best_fit = 0.6 * self.current_fit + 0.4 * avg_prev_fits
            self.best_fit = best_fit

    def get_best_fit(self):
        if self.best_fit is None:
            return self.current_fit
        return self.best_fit

    # set line curvance
    def set_curvance(self, curvature):
        self.radius_of_curvature = curvature 
    # get line curvance
    def get_curvance(self):
        return self.radius_of_curvature
    # set the distance from vehicle to the line     
    def set_line_base_pos(self, distance):
        self.line_base_pos = distance 
    # get the distance from vehicle to the line     
    def get_line_base_pos(self):
        return self.line_base_pos 
    
    # Update history fit
    def _update_history_fits(self, current_fit):
        if current_fit is None:
            return
        # idx 0 hold the most recent fit 
        for i in range(self.history_fits.shape[0]-1, 0,-1):
            self.history_fits[i] = self.history_fits[i-1]
        self.history_fits[0] = current_fit

    def update_fit(self, current_fit):
        self.detected  = True if current_fit is not None else False
        if self.detected:
            self._update_history_fits(current_fit)
            self.calculate_best_fit()

class LaneDetector(object):
    """
    Lane detector object
    """
    def __init__(self, in_file, out_path=None, dbg_out_path = None):
        self.in_file       = in_file
        self.out_path     = out_path
        self.dbg_out_path = dbg_out_path

        if out_path and not os.path.exists(out_path):
            os.makedirs(out_path)
        if dbg_out_path and not os.path.exists(dbg_out_path):
            os.makedirs(dbg_out_path)
        print(in_file, out_path, dbg_out_path)
        self.left_line  = Line()
        self.right_line = Line()
        # Perspective transform source points
        self.ppt_src    = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
        # self.ppt_src    = np.float32([[(200, 720), (645, 470), (720, 470), (1130, 720)]])
        # Perspective transform des points
        self.ppt_dst    = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])
        # Perspective transform matrix 
        self.PPT_M      = cv2.getPerspectiveTransform(self.ppt_src, self.ppt_dst)
        # Perspective transform invert matrix 
        self.PPT_M_inv  = cv2.getPerspectiveTransform(self.ppt_dst, self.ppt_src)
        # Relation between pixel space and real world space 
        self.ym_per_pix = None
        self.xm_per_pix = None
        # image dimentions 
        self.img_width  = None
        self.img_height = None
        self.processed_count = 0

    def _binarize_image(self, image):
        return binarize_image(image)

    def _perspective_transform(self, image):
        img_size = image.shape[:2][::-1]
        return cv2.warpPerspective(image, self.PPT_M, img_size , flags=cv2.INTER_LINEAR)

    def sliding_window_search(self, warped_image, is_video):        
        reset_sliding_window = True
        left_fit = None
        right_fit = None

        prev_left_cur  = self.left_line.get_curvance()
        prev_right_cur = self.right_line.get_curvance()
        
        # if both left and right lines have been detected
        # use previous left/right fit to speed up the search 
        if self.left_line.detected and self.right_line.detected and is_video:
            left_fit, right_fit = sliding_window_use_prev_fits(warped_image, self.left_line.current_fit, self.right_line.current_fit)
            # sanity check the search results with existing fit 
            if left_fit is None or right_fit is None:
                print('error: left_fit %s, right_fit %s ' % (left_fit, right_fit))
            else:
                left_cur, right_cur, vehicle_pos = self._calculate_curvance(left_fit, right_fit)
                left_cur_change = left_cur / prev_left_cur
                right_cur_change = right_cur / prev_right_cur
                left_change_too_large = left_cur_change > 10.0 or left_cur_change < 0.1
                right_change_too_large = right_cur_change > 10.0 or left_cur_change < 0.1
                # if the curvance is too large or vehicle_pos is larger than 2 meter, reset sliding window search 
                if left_change_too_large or right_change_too_large or vehicle_pos > 2:
                    print('warning: resetting sliding window. cur:[%s, %s], dis: [%s]' %(left_cur, right_cur, vehicle_pos))
                else:
                    reset_sliding_window = False

        if reset_sliding_window:
            left_fit, right_fit = sliding_window_search(warped_image)
        
        left_cur  = -1.0
        right_cur = -1.0
        vehicle_pos  = -1.0

        self.left_line.current_fit = left_fit
        self.right_line.current_fit = right_fit

        # store current fit results for video processing
        if is_video:
            self.left_line.update_fit(left_fit)
            self.right_line.update_fit(right_fit)
 
        left_fit = self.left_line.get_best_fit()
        right_fit = self.right_line.get_best_fit()
        
        if left_fit is not None and right_fit is not None:
            left_cur, right_cur, vehicle_pos = self._calculate_curvance(left_fit, right_fit)
        self.left_line.set_curvance(left_cur)
        self.right_line.set_curvance(right_cur)
        self.left_line.set_line_base_pos(vehicle_pos)
        self.right_line.set_line_base_pos(vehicle_pos)

    def _calculate_curvance(self, left_fit, right_fit):
        ploty      = np.linspace(0, self.img_height-1, self.img_height)
        left_fitx  = left_fit[0]*ploty**2  + left_fit[1]*ploty  + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        y_eval     = np.median(ploty)
        
        leftx      = left_fitx[::-1]  # Reverse to match top-to-bottom in y
        rightx     = right_fitx[::-1] # Reverse to match top-to-bottom in y

        ym_per_pix = self.ym_per_pix # meters per pixel in y dimension
        xm_per_pix = self.xm_per_pix # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr    = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr   = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Calculate the vehicle position    
        vehicle_position = abs(640 - ((left_fitx[-1]+right_fitx[-1])/2)) * xm_per_pix
    
        return left_curverad, right_curverad, vehicle_position

    def draw_detected_lanes(self, image):
        left_fit  = self.left_line.get_best_fit()
        right_fit = self.right_line.get_best_fit()
        
        ploty      = np.linspace(0, self.img_height-1, self.img_height)
        left_fitx  = left_fit[0]*ploty**2  + left_fit[1]*ploty  + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        color_warp = np.zeros_like(image).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts       = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.PPT_M_inv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

    def _process_image(self, frame, is_video, name_hint = None):
        self.img_width  = frame.shape[1]
        self.img_height = frame.shape[0]

        self.ym_per_pix = 30.0/frame.shape[0]
        self.xm_per_pix = 3.7/frame.shape[1]

        # convert the image to binary, only keep the most interesting part
        binary_image = self._binarize_image(frame)
        warped_image = None

        if binary_image is not None:
            warped_image = self._perspective_transform(binary_image)
    
        self.sliding_window_search(warped_image, is_video)
        out_frame = self.draw_detected_lanes(frame)
        # Put radius of curvature info on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        radius_text = "radius of curvature - left %s m, right %s m" % (round(self.left_line.get_curvance(), 2), round(self.right_line.get_curvance(), 2))
        cv2.putText(out_frame, radius_text,(100, 30), font, 1,(255, 255, 255), 2)
        # Put vehicle positon info on the image
        pos_text = "vehicle position is %sm left of center" % (round(self.left_line.get_line_base_pos(), 2))
        cv2.putText(out_frame, pos_text,(100, 60), font, 1,(255, 255, 255), 2)
            
        # Store the temporary processing results for debugging purpose 
        if self.dbg_out_path:
            image_name = name_hint
            if image_name is None:
                image_name =  "frame_%d.jpg" % self.processed_count
            # cv2.imwrite(self.dbg_out_path + 'orginal_'   + image_name, frame) 
            cv2.imwrite(self.dbg_out_path + 'binary_'    + image_name, binary_image*255) 
            cv2.imwrite(self.dbg_out_path + 'warped_'    + image_name, warped_image*255) 
            cv2.imwrite(self.dbg_out_path + 'processed_' + image_name, out_frame)
            print(image_name, self.left_line.get_curvance(), 'm', self.right_line.get_curvance(), 'm', self.right_line.get_line_base_pos(), 'm')
        self.processed_count +=1
        return out_frame
    def process_image(self):
        print('processing ' + str(self.in_file))
        image = cv2.imread(self.in_file)
        image_name = self.in_file.split('/')[-1]
        self._process_image(image, False, image_name)

    def process_video(self):
        count = 0
        video_cap = cv2.VideoCapture(self.in_file)
        fourcc = cv2.cv.CV_FOURCC(*'DIVX')
        is_writer_inited = False
        while(video_cap.isOpened()):
            # read a image from video file
            ret, frame = video_cap.read()
            if not is_writer_inited:
                out_video = cv2.VideoWriter(self.out_path + 'output.avi',fourcc, 30.0, frame.shape[:2][::-1])
                is_writer_inited = True
            if frame is None:
                print('no more video frame')
                break
            out_frame = self._process_image(frame, True)
            out_video.write(out_frame)

# Process a video file 
ld = LaneDetector('project_video.mp4', 'out_video/', dbg_out_path=None)
ld.process_video()

# # Process all the example image file 
img_path  = 'output_images/undistorted/'
img_files = glob.glob(img_path + '*jpg') +  glob.glob(img_path + '*.png')
for img_file in img_files:
    ld = LaneDetector(img_file, 'output_images/pipeline_out/', dbg_out_path='output_images/pipeline_out/')
    ld.process_image()



