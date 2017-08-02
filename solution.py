import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob 
import os 
import pickle


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Select while and yellow from rgb space 
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([180, 180, 180])
    upper = np.uint8([255, 255, 255])
    white_color_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([180, 180,   0])
    upper = np.uint8([255, 255, 255])
    yellow_color_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    color_mask = cv2.bitwise_or(white_color_mask, yellow_color_mask)
    color_select = cv2.bitwise_and(image, image, mask = color_mask)
    return color_select

# Select while and yellow from hls space 
def select_hls_white_yellow(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hls, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(hls, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


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


# Edit this function to create your own pipeline.
def process_image(img, sx_thresh=(10, 250)):
    img  = np.copy(img)
    img = gaussian_blur(img, 5)
    color_select = select_hls_white_yellow(img)
    # Convert to gray image 
    gray = cv2.cvtColor(color_select, cv2.COLOR_RGB2GRAY)
    ret,gray_binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(color_select, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:,:,2]

    # Sobel x
    sobelx       = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx   = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(gray_binary > 0) | (sxbinary > 0)] = 1

    imshape = img.shape
    vertices = np.array([[ (75,imshape[0]), (450, 450), (775, 450),(1200,imshape[0])]], dtype=np.int32)  
    combined_binary = region_of_interest(sxbinary, vertices)

    return combined_binary

img_path1 = 'output_images/test_images_undistored/*jpg'
img_path2 = 'output_images/test_images_undistored/*png'
img_files = glob.glob(img_path1)


for img_file in img_files:
    # Load color images into BGR format 
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    thresholded = process_image(image)

    bottom_left = [320,720] 
    bottom_right = [920, 720]
    top_left = [320, 1]
    top_right = [920, 1]

    dst   = np.float32([bottom_left,bottom_right,top_right,top_left])

    src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
    dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])

    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    img_size = image.shape[:2][::-1]

    warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)


    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    name = img_file.split('/')[-1]
    ax1.set_title(name, fontsize=40)
    
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('warped', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

if 0:
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML


    input_folder = './' 
    output_folder = 'test_videos_output/' 
    video_file = 'project_video.mp4'
    processed_file = os.path.join(output_folder, video_file)

    clip = VideoFileClip(os.path.join(input_folder, video_file))
    white_clip = clip.fl_image(process_image) #NOTE: this function expects color images!! 
    white_clip.write_videofile(processed_file, audio=False)
    HTML("""
    <video width="1280" height="720" controls>
    <source src="{0}">
    </video>
    """.format(processed_file))
