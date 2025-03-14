#####################################
"""
def threshold(channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
    Applies a threshold to an image channel.
    Parameters:
        channel: The image channel to be thresholded.
        thresh: Tuple specifying lower and upper threshold values.
        thresh_type: OpenCV thresholding type (e.g., cv2.THRESH_BINARY).
    Returns:
        Thresholded image.
    
    
def binary_array(array, thresh, value=0): 
    Converts an array into a binary representation.
    Parameters:
        array: Input array to apply the binary threshold.
        thresh: Tuple with lower and upper threshold values.
        value: Value to assign to pixels within the threshold range.
    Returns:
        Binary array with values set accordingly.
    
    
def sobel(img_channel, orient='x', sobel_kernel=3): 
    Applies the Sobel edge detection filter.
    Parameters:
        img_channel: Single-channel image to apply the Sobel operator.
        orient: 'x' for vertical edges, 'y' for horizontal edges.
        sobel_kernel: Kernel size for the Sobel operator.
    Returns:
        Sobel-filtered image.
    
    
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    Computes the gradient magnitude and applies a threshold.
    Parameters:
        image: Input image for gradient magnitude thresholding.
        sobel_kernel: Kernel size for the Sobel operator.
        thresh: Tuple defining lower and upper threshold values.
    Returns:
        Binary image after applying magnitude threshold.
    
    
def transform(img, timestamp, x): 
    Applies a perspective transformation.
    Parameters:
        img: Input image for perspective transformation.
        timestamp: Time information for transformation calculations.
        x: Direction of transformation (1 for forward, -1 for inverse).
    Returns:
        Warped image based on transformation.
    
    
def threshbinary(img, timestamp, white_thresh=240, new_val=200):
    Converts an image to a binary format by thresholding bright pixels.
    Parameters:
        img: Input image.
        timestamp: Time-based transformation parameter.
        white_thresh: Threshold value for white pixels.
        new_val: New value for binary transformation.
    Returns:
        Binary image with threshold applied.
    
    
def pfit(binary_img, nwindows=50, margin=40, minpix=20):
    Uses a sliding window approach to fit a polynomial lane curve.
    Parameters:
        binary_img: Binary warped image.
        nwindows: Number of sliding windows.
        margin: Width of each window.
        minpix: Minimum number of pixels to recenter window.
    Returns:
        left_fit: Polynomial fit coefficients for the left lane.
        right_fit: Polynomial fit coefficients for the right lane.
    
    
def draw(original_img, binary_img, left_fit, right_fit, timestamp): 
    Draws detected lane lines on the original image.
    Parameters:
        original_img: Original frame.
        binary_img: Binary processed frame.
        left_fit: Left lane polynomial fit.
        right_fit: Right lane polynomial fit.
        timestamp: Time-based transformation parameter.
    Returns:
        Image with lane lines drawn.
    
    
def proc_img(img, timestamp):  
    Processes an image to detect lane markings.
    Parameters:
        img: Input image.
        timestamp: Time-based transformation parameter.
    Returns:
        Processed image with detected lanes.
    
    
class Lane:  
    Class to handle lane detection data and processing.
    Parameters:
        orig_frame: Original frame for lane detection.
    Returns:
        Lane object with attributes like warped_frame and transformation matrix.  
        
class GUI: 
    Creates a graphical user interface (GUI) for lane detection.
    Parameters:
        root: Root window for the GUI.
    Returns:
        GUI instance with video display and controls.
"""
#####################################




import cv2
import numpy as np
import sys
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt




def matplotlib_display(img, title='Image'):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def timeROI(timestamp):
    if timestamp < 20:
        return [[580, 965], [115, 1210], [1074, 1181], [742, 963]]
        #return [[280, 965], [115, 1210], [1074, 1181], [742, 963]]
    elif timestamp < 69:
        return [[480, 965], [115, 1210], [1074, 1181], [742, 963]]
    else:
        return [[280, 965], [115, 1210], [1074, 1181], [742, 963]]


def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY):
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)


def binary_array(array, thresh, value=0):
  if value == 0:
    binary = np.ones_like(array)
  else:
    binary = np.zeros_like(array)  
    value = 1
  binary[(array >= thresh[0]) & (array <= thresh[1])] = value
  return binary


def sobel(img_channel, orient='x', sobel_kernel=3):
  if orient == 'x':
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
  if orient == 'y':
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)
  return sobel


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
  sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
  sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))
  mag = np.sqrt(sobelx ** 2 + sobely ** 2)
  return binary_array(mag, thresh)


def transform(img,timestamp,x):
    output = img.copy()
    h, w = output.shape[:2]
    padding = int(0.25 * w) # padding from side of the image in pixels
    droi = np.array([
      [padding, 0], # Top-left corner
      [padding, h], # Bottom-left corner        
      [w-padding, h], # Bottom-right corner
      [w-padding, 0] # Top-right corner
    ], dtype=np.float32)  
    tl, bl, br, tr = timeROI(timestamp)
    roi = np.array([tl, bl, br, tr], dtype=np.float32)
    tmatrix = cv2.getPerspectiveTransform(roi,droi)        
    invtmatrix = cv2.getPerspectiveTransform(droi,roi)
    if x==1:
        return cv2.warpPerspective(output, tmatrix, (w,h), flags=(cv2.INTER_LINEAR))
    if x == -1:
        return cv2.warpPerspective(output, invtmatrix, (w,h), flags=(cv2.INTER_LINEAR))




def threshbinary(img, timestamp, white_thresh=240, new_val=200):
    output = img.copy()
    hls = cv2.cvtColor(output, cv2.COLOR_BGR2HLS)
    _, sxbinary = threshold(hls[:, :, 2], thresh=(80, 255))
    sxbinary = mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
    gray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,215,255,cv2.THRESH_BINARY)
    img = cv2.bitwise_or(thresh,sxbinary.astype(np.uint8))
    warped_frame = transform(img,timestamp,1)
    thresh, binary_warped = cv2.threshold(warped_frame, 215, 255, cv2.THRESH_BINARY)          
    return binary_warped






def pfit(binary_img, nwindows=50, margin=40, minpix=20):
    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = int(binary_img.shape[0] // nwindows)
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        cv2.rectangle(binary_img, (leftx_current, win_y_high), (rightx_current, win_y_low), (255, 0, 0), 5)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else []
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else []
    leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else []
    lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else []
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else []
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else []
    left_fit = None
    right_fit = None
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def draw(original_img, binary_img, left_fit, right_fit, timestamp):
    rarrow = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)
    if rarrow is not None:
        rarrow = cv2.resize(rarrow, (250,250))
    rarrow = cv2.resize(rarrow, (250,250))
    result = original_img.copy()
    overlay = np.zeros_like(original_img)
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    if left_fit is None and right_fit is not None:
        right_fit_mirror = right_fit
        left_fit = np.array([-right_fit_mirror[0], -right_fit_mirror[1], original_img.shape[1] - right_fit_mirror[2]])
    elif right_fit is None and left_fit is not None:
        left_fit_mirror = left_fit
        right_fit = np.array([-left_fit_mirror[0], -left_fit_mirror[1], original_img.shape[1] - left_fit_mirror[2]])
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        pts_center = np.array([np.transpose(np.vstack([(left_fitx + right_fitx) / 2.0, ploty]))])
        cv2.polylines(overlay, np.int32(pts_left), False, (0, 0, 255), 5)
        cv2.polylines(overlay, np.int32(pts_right), False, (255, 0, 0), 5)
        cv2.polylines(overlay, np.int32(pts_center), False, (0, 255, 0), 3)
        center_fit = (left_fit + right_fit) / 2.0
        y_tip = 10
        delta = 10
        x_tip = center_fit[0]*(y_tip**2) + center_fit[1]*y_tip + center_fit[2]
        y_next = y_tip + delta
        x_next = center_fit[0]*(y_next**2) + center_fit[1]*y_next + center_fit[2]
        dx = x_next - x_tip
        dy = y_next - y_tip
        angle_rad = math.atan2(dx, dy)
        arrow_angle = math.degrees(angle_rad)
    overlay = transform(overlay,timestamp,-1)
    result = cv2.addWeighted(overlay, 2, original_img, 0.5, 0)
    if rarrow is not None and left_fit is not None and right_fit is not None:
        (h, w) = rarrow.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, arrow_angle, 1.0)
        rarrow_rotated = cv2.warpAffine(rarrow, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        arrow_h, arrow_w = rarrow_rotated.shape[:2]
        x_offset = result.shape[1] - arrow_w
        y_offset = 0
        if rarrow_rotated.shape[2] == 4:
            arrow_rgb = rarrow_rotated[:, :, :3]
            arrow_alpha = rarrow_rotated[:, :, 3] / 255.0
            roi = result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w]
            for c in range(3):
                roi[:, :, c] = arrow_alpha * arrow_rgb[:, :, c] + (1 - arrow_alpha) * roi[:, :, c]
            result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w] = roi
        else:
            result[y_offset:y_offset+arrow_h, x_offset:x_offset+arrow_w] = rarrow_rotated
    return result


def proc_img(img, timestamp):
    img_modified = threshbinary(img, timestamp, white_thresh=240, new_val=200)
    left_fit, right_fit = pfit(img_modified)
    if left_fit is None and right_fit is None:
        print("Warning: No lane pixels detected.")
        return img_modified
    result = draw(img, img_modified, left_fit, right_fit, timestamp)
    #tl, bl, br, tr = timeROI(timestamp)
    #points = np.array([tl, bl, br, tr], dtype=np.int32)
    #cv2.polylines(result, [points], isClosed = True, color = (255, 255, 255), thickness = 5)
    return result


class Lane:


  def __init__(self, orig_frame):


    self.orig_frame = orig_frame
 
    # This will hold an image with the lane lines      
    self.lane_line_markings = None
 
    # This will hold the image after perspective transformation
    self.warped_frame = None
    self.transformation_matrix = None
    self.inv_transformation_matrix = None
 
    # (Width, Height) of the original video frame (or image)
    self.orig_image_size = self.orig_frame.shape[::-1][1:]
 
    width = self.orig_image_size[0]
    height = self.orig_image_size[1]
    self.width = width
    self.height = height
     
    # Four corners of the trapezoid-shaped region of interest
    # You need to find these corners manually.
    self.roi_points = np.float32([
      (580,965), # Top-left corner
      (115,1210), # Bottom-left corner            
      (1074,1181), # Bottom-right corner
      (742,963) # Top-right corner
    ])
         
    # The desired corner locations  of the region of interest
    # after we perform perspective transformation.
    # Assume image width of 600, padding == 150.
    self.padding = int(0.25 * width) # padding from side of the image in pixels
    self.desired_roi_points = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.orig_image_size[1]], # Bottom-left corner        
      [self.orig_image_size[
        0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
      [self.orig_image_size[0]-self.padding, 0] # Top-right corner
    ])








class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUIGUIGUI")
        self.video_width = 600
        self.video_height = 450
        self.streaming = False
        self.cap = None
        self.placeholder = self.create_placeholder(self.video_width, self.video_height)
        self.raw_label = tk.Label(root, image=self.placeholder, width=self.video_width, height=self.video_height)
        self.raw_label.grid(row=0, column=0)
        self.proc_label = tk.Label(root, image=self.placeholder, width=self.video_width, height=self.video_height)
        self.proc_label.grid(row=1, column=0)
        self.ctrl_frame = tk.Frame(root, width=self.video_width, height=self.video_height)
        self.ctrl_frame.grid(row=0, column=1, sticky="nsew")
        self.ctrl_frame.grid_columnconfigure(0, weight=1)
        self.ctrl_frame.grid_columnconfigure(1, weight=1)
        self.ctrl_frame.grid_columnconfigure(2, weight=1)
        self.ctrl_frame.grid_rowconfigure(0, weight=1)
        self.ctrl_frame.grid_rowconfigure(1, weight=1)
        self.ctrl_frame.grid_rowconfigure(2, weight=1)
        self.up_button = tk.Button(self.ctrl_frame, text="Up", width=10, command=self.move_up)
        self.left_button = tk.Button(self.ctrl_frame, text="Left", width=10, command=self.move_left)
        self.center_button = tk.Button(self.ctrl_frame, text="Start", width=10, command=self.toggle_stream)
        self.right_button = tk.Button(self.ctrl_frame, text="Right", width=10, command=self.move_right)
        self.down_button = tk.Button(self.ctrl_frame, text="Down", width=10, command=self.move_down)
        self.up_button.grid(row=0, column=1, padx=5, pady=5)
        self.left_button.grid(row=1, column=0, padx=5, pady=5)
        self.center_button.grid(row=1, column=1, padx=5, pady=5)
        self.right_button.grid(row=1, column=2, padx=5, pady=5)
        self.down_button.grid(row=2, column=1, padx=5, pady=5)
        self.up_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.left_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.right_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.down_button.bind("<ButtonRelease-1>", lambda e: self.add_log("stop moving"))
        self.log_text = tk.Text(root, height=20, width=50)
        self.log_text.grid(row=1, column=1, sticky="nsew")
    def create_placeholder(self, width, height, color=(0, 0, 0)):
        from PIL import Image
        image = Image.new('RGB', (width, height), color)
        return ImageTk.PhotoImage(image)
    def add_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    def move_up(self):
        self.add_log("moving up")
    def move_down(self):
        self.add_log("moving down")
    def move_left(self):
        self.add_log("moving left")
    def move_right(self):
        self.add_log("moving right")
    def toggle_stream(self):
        if self.streaming:
            self.streaming = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.center_button.config(text="Start")


            self.update_placeholder()
        else:
            self.cap = cv2.VideoCapture("Dylan.mov")
            self.streaming = True
            self.center_button.config(text="Stop")


            self.update_frame()
    def update_placeholder(self):
        self.raw_label.config(image=self.placeholder)
        self.raw_label.image = self.placeholder
        self.proc_label.config(image=self.placeholder)
        self.proc_label.image = self.placeholder
    def update_frame(self):
        if self.streaming and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                print(timestamp)
                raw_frame = frame
                (h, w) = raw_frame.shape[:2]
                new_width = 300
                aspect_ratio = h / w
                new_height = int(new_width * aspect_ratio)
                raw_frame = cv2.resize(frame, (new_width, new_height))
                raw_image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                # Unpack ROI coordinates
                raw_image = Image.fromarray(raw_image)
                raw_photo = ImageTk.PhotoImage(raw_image)
                self.raw_label.config(image=raw_photo)
                self.raw_label.image = raw_photo
                proc_frame = proc_img(frame, timestamp)
                proc_frame = cv2.resize(proc_frame, (new_width, new_height))
                proc_image = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                proc_image = Image.fromarray(proc_image)
                proc_photo = ImageTk.PhotoImage(proc_image)
                self.proc_label.config(image=proc_photo)
                self.proc_label.image = proc_photo
            else:
                self.add_log("Out of frames")
                self.streaming = False
                self.center_button.config(text="Start")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.update_placeholder()
            self.root.after(30, self.update_frame)
        else:
            self.update_placeholder()


def main():
    cap = cv2.VideoCapture("Dylan.mov")
    while cap.isOpened():
        ret, frame = cap.read()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()


