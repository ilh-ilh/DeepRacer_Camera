#!/usr/bin/env python3

# ros1으로부터 ros2로 포팅된 코드
import rclpy
import cv2
import numpy as np

from rclpy.qos import QoSProfile
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from sensor_msgs.msg import Image  # CompressedImage 대신 Image 임포트

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.bridge = CvBridge()
        qos = QoSProfile(depth=10)
        
        # 구독하는 메시지 타입을 Image로 변경
        self.subscription = self.create_subscription(Image, "/topic_name", self.image_callback, qos)
        self.pub_steer = self.create_publisher(Float64, "/lane_steer", qos)
        
        self.left_fit_prev = None
        self.right_fit_prev = None
        self.prev_steering_angle = None

        self.stability_threshold = 10
        self.smooth_factor = 0.8

    def _validate_and_use_prev_fit(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit_prev[0] * (nonzeroy ** 2) +
                                    self.left_fit_prev[1] * nonzeroy + self.left_fit_prev[2] - margin)) &
                        (nonzerox < (self.left_fit_prev[0] * (nonzeroy ** 2) +
                                    self.left_fit_prev[1] * nonzeroy + self.left_fit_prev[2] + margin)))

        right_lane_inds = ((nonzerox > (self.right_fit_prev[0] * (nonzeroy ** 2) +
                                        self.right_fit_prev[1] * nonzeroy + self.right_fit_prev[2] - margin)) &
                        (nonzerox < (self.right_fit_prev[0] * (nonzeroy ** 2) +
                                        self.right_fit_prev[1] * nonzeroy + self.right_fit_prev[2] + margin)))

        leftx, rightx = nonzerox[left_lane_inds], nonzerox[right_lane_inds]
        if len(leftx) > 0 and len(rightx) > 0:
            lane_gap = np.mean(rightx) - np.mean(leftx)
            if np.abs(lane_gap - 300) < self.stability_threshold:
                left_fit = np.polyfit(nonzeroy[left_lane_inds], leftx, 2)
                right_fit = np.polyfit(nonzeroy[right_lane_inds], rightx, 2)
                return left_fit, right_fit, True
            
        return None, None, False
            
    def image_callback(self, msg):
        # sensor_msgs/Image 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 차선 인식 프로세스 실행
        result, steering_angle = self.process_frame(frame)
        self.pub_steer.publish(steering_angle)
        cv2.imshow("Lane Detection", result)
        self.get_logger().info(f"Steering Angle: {steering_angle:.2f} degrees")
        cv2.waitKey(1)

    def process_frame(self, frame):
        # 이미지 크기 조정
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sobel Edge Detection
        sobelx = self.sobel_xy(gray, orient='x', thresh=(20, 100))
        sobely = self.sobel_xy(gray, orient='y', thresh=(20, 100))
        gradient_mag = self.gradient_magnitude(gray, thresh=(30, 255))

        # HLS 색상 공간 변환 및 S 채널 추출
        hls_s = self.hls_select(frame, thresh_white=(90, 255), thresh_yellow_orange=(15, 40))

        # Edge 및 색상 정보 결합
        combined = np.zeros_like(sobelx)
        combined[((sobelx == 255) & (sobely == 255)) | ((gradient_mag == 255) & (hls_s == 255))] = 255

        # 원근 변환
        warped, src, dst = self.perspective_transform(combined)

        # 적응형 차선 탐색
        left_fit, right_fit = self.adaptive_search(warped)

        # 차선 기반 조향각 계산
        steering_angle = self.calculate_steering_angle(left_fit, right_fit, warped.shape)

        # 원본 이미지에 차선 그리기
        result = self.draw_lane(frame, warped, left_fit, right_fit, src, dst)
        return result, steering_angle
    
    def sobel_xy(self, img, orient='x', thresh=(20, 100)):
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
        return binary_output
    
    def gradient_magnitude(self, img, sobel_kernel=3, thresh=(30, 255)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 255
        return binary_output

    def hls_select(self, img, thresh_white=(90, 255), thresh_yellow_orange=(15, 40)):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h_channel = hls[:, :, 0]
        s_channel = hls[:, :, 2]

        white_binary = np.zeros_like(s_channel)
        white_binary[(s_channel >= thresh_white[0]) & (s_channel <= thresh_white[1])] = 255

        yellow_orange_binary = np.zeros_like(h_channel)
        yellow_orange_binary[(h_channel >= thresh_yellow_orange[0]) & (h_channel <= thresh_yellow_orange[1]) & (s_channel >= 100)] = 255

        combined_binary = cv2.bitwise_or(white_binary, yellow_orange_binary)
        return combined_binary

    def perspective_transform(self, img):
        height, width = img.shape[:2]
        src = np.float32([[125, height], [550, height], [300, 225], [400, 225]])
        dst = np.float32([[100, height], [540, height], [100, 0], [540, 0]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
        return warped, src, dst

    def adaptive_search(self, binary_warped):
        if self.left_fit_prev is not None and self.right_fit_prev is not None:
            left_fit, right_fit, reliable = self._validate_and_use_prev_fit(binary_warped)

            if not reliable:
                left_fit, right_fit = self.sliding_window(binary_warped)
        else:
            left_fit, right_fit = self.sliding_window(binary_warped)

        self.left_fit_prev = left_fit
        self.right_fit_prev = right_fit
        return left_fit, right_fit
    
    def calculate_steering_angle(self, left_fit, right_fit, img_shape):
        y_eval = img_shape[0]
        left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        lane_center = (left_x + right_x) / 2
        camera_center = img_shape[1] / 2
        offset = lane_center - camera_center
        angle = np.arctan2(offset, y_eval)
        new_steering_angle = np.degrees(angle)

        if self.prev_steering_angle is not None:
            steering_angle = (self.smooth_factor * self.prev_steering_angle +
                              (1 - self.smooth_factor) * new_steering_angle)
        else:
            steering_angle = new_steering_angle

        self.prev_steering_angle = steering_angle
        return steering_angle
    
    def sliding_window(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        n_windows = 9
        window_height = binary_warped.shape[0] // n_windows
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = nonzero[0], nonzero[1]
        leftx_current, rightx_current = leftx_base, rightx_base
        margin, minpix = 50, 25
        left_lane_inds, right_lane_inds = [], []

        for window in range(n_windows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) else self.left_fit_prev
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) else self.right_fit_prev

        return left_fit, right_fit
    
    def draw_lane(self, original_img, binary_img, left_fit, right_fit, src, dst):
        height, width = binary_img.shape
        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        M_inv = cv2.getPerspectiveTransform(dst, src)
        newwarp = cv2.warpPerspective(color_warp, M_inv, (width, height))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
        return result

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
