#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


bridge = CvBridge()
cv_image = np.empty(shape=[0])
motor = None

initialized = False
last_x_location = 279
x_loc = 0
flag = False

CAM_FPS = 30    # 카메라 FPS - 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기

class Warper:
    def __init__(self):
	# img의 가로 / 세로
        h = 480
        w = 640
        print("h : " ,h)
        print("w : " ,w)
         
        # distort src(INPUT) to dst(OUTPUT) 
        src = np.float32([ # 4개의 원본 좌표 점
            [w * 1.5, h * 1.3], # [960, 624]
            [w * (-0.1), h * 1.3], # [-64.0, 624]
            [0, h * 0.62], # [0, 297.6]
            [w, h * 0.62], # [640, 297.6]
        ])
        dst = np.float32([ # 4개의 결과 좌표 점 - 투시 변환 후의 이미지에서 'src' 좌표가 매핑되는 위치를 나타냄.
            [w * 0.65, h * 0.98], # [416, 470.4]
            [w * 0.35, h * 0.98], # [224, 470.4]
            [w * (-0.3), 0], # [-192, 0]
            [w * 1.3, 0], # [832, 0]
        ])
        

        self.M = cv2.getPerspectiveTransform(src, dst) # self.M : 투시변환 행렬(src to dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src) # self.Minv : 투시변환 행렬(dst to src)

    # 이미지를 투시 변환하여 변형된 이미지를 반환 - img -> self.M 투시 변환 행렬을 적용하여 변형 -> 변형된 이미지의 크기는 원본 이미지의 가로(img.shape[1])와 세로(img.shape[0]) 크기로 설정
    def warp(self, img): 
        return cv2.warpPerspective(
            img,
            self.M, 
            (img.shape[1], img.shape[0]), # img w, h
            flags=cv2.INTER_LINEAR # 이미지 보간 방법을 선형 보간으로 설정
        )
    # 이미지를 역 투시 변환하여 원본 이미지로 복원 - img -> self.Minv 역 투시 변환 행렬을 적용하여 원본 이미지로 역변환 -> 원본 이미지의 가로(img.shape[1])와 세로(img.shape[0]) 크기
    def unwarp(self, img):
        return cv2.warpPersective(
            img,
            self.Minv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR # 이미지 보간 방법을 선형 보간으로 설정
        )

#=============================================
#sliding window를 위한 클래스
#=============================================

TOTAL_CNT = 50

class SlideWindow:
    
    def __init__(self):
        self.current_line = "DEFAULT" #현재 라인의 문자열 값
        self.left_fit = None #왼쪽 측면의 픽셀 값에 대한 회귀 모델의 계수
        self.right_fit = None #오른쪽 측면의 픽셀 값에 대한 회귀 모델의 계수
        self.leftx = None #왼쪽 측면의 픽셀 위치를 나타내는 변수
        self.rightx = None #오른쪽 측면의 픽셀 위치를 나타내는 변수
        self.left_cnt = 25 #왼쪽 측면의 count값
        self.right_cnt = 25 #오른쪽 측면의 count값

    def slidewindow(self, img, flag):
	    #x_location은 none으로 설정해줌.
        x_location = None
        #np.dstack배열을 차원별로 수평으로 쌓아 새로운 배열을 생성 - 배열의 깊이 방향으로 쌓는다. - img 배열을 3번 반복하여 수평으로 쌓아 새로운 3차원 배열을 생성 -> 3개의 채널이 모두 img 배열의 값으로 채워진 RGB 이미지가 생성 -> 연산을 통해 배열의 값을 255로 곱하여 픽셀 값 범위를 0-255로 조정 
	    #이는 대부분의 이미지 표현 방식에서 사용되는 범위로 픽셀 값을 조정      
        out_img = np.dstack((img, img, img)) * 255 # deleted 
	    # out_img를 다시 img로 재선언
        out_img = img 
	    # img.shpae = 이미지의 크기를 반환하는 함수 
	    # img.shape[0] = 이미지의 높이 = 이미지의 픽셀의 행 개수
        height = img.shape[0]
	    # img.shape[1] = 이미지의 너비 = 이미지의 픽셀의 열 개수
        width = img.shape[1]

        # 윈도우의 높이 = 7 & 윈도우의 개수 = 25
        window_height = 7 
        nwindows = 30 
        
	    # img배열에서 0이 아닌 요소들의 인덱스 반환 
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0]) # nonzero 배열 y좌표
        nonzerox = np.array(nonzero[1]) # nonzero 배열 x좌표
        # sliding window를 하기 위해 필요한 변수들
        margin = 20  #차선 주변의 픽세을 검색하는 값
        minpix = 10 #윈도우 내에서 검출되어야 하는 최소 픽셀 수
        left_lane_inds = [] #왼쪽 차선 픽셀 인덱스 저장하는 리스트
        right_lane_inds = [] #오른쪽 차선 픽셀 인덱스 저장하는 리스트
	    #각각 윈도우 영역의 높이와 너비를 지정하는 변수
        win_h1 = 340
        win_h2 = 480
        win_l_w_l = 125
        win_l_w_r = 175
        win_r_w_l = 445
        win_r_w_r = 495

        if flag == None :
            win_l_w_r = 250
            win_r_w_l = 350

        # 템프
        temp = 0.38
        half_temp = 0.5 * temp
        
	    # 위치를 찾고 위치 분리
	    # pts_left = 왼쪽 차선 영역을 나타내는 다각형의 꼭지점 좌표 - 차선을 파란색으로 칠함.
	    # pts_left - win_l_w_l은 왼쪽 변의 아래 꼭지점의 x 좌표, win_l_w_r은 왼쪽 변의 위 꼭지점의 x 좌표, win_h1은 아래 변의 y 좌표, win_h2는 위 변의 y 좌표
        pts_left = np.array([[win_l_w_l, win_h2], [win_l_w_l, win_h1], [win_l_w_r, win_h1], [win_l_w_r, win_h2]], np.int32)
	    # 다각형을 그릴 때는 cv2.polylines 메서드를 이용
	    # 다각형을 구성하는 점들의 배열을 리스트 = pts_left를 배열로 받음. -> False로 설정하면 열려있는 다각형 -> 파란색을 나타내는 (255, 0, 0)을 사용 -> 다각형의 선 두께를 나타내는 정수 값 = 1     
        cv2.polylines(out_img, [pts_left], False, (255,0,0), 1)
	    # pts_right = 오른쪽 차선 영역을 나타내는 다각형의 꼭지점 좌표 - 차선을 파란색으로 칠함.
	    # win_r_w_l은 오른쪽 변의 아래 꼭지점의 x 좌표, win_r_w_r은 오른쪽 변의 위 꼭지점의 x 좌표, win_h1은 아래 변의 y 좌표, win_h2는 위 변의 y 좌표
        pts_right = np.array([[win_r_w_l, win_h2], [win_r_w_l, win_h1], [win_r_w_r, win_h1], [win_r_w_r, win_h2]], np.int32)
	    # 다각형을 구성하는 점들의 배열을 리스트 = pts_right를 배열로 받음 -> False로 설정하면 열려있는 다각형 -> 파란색을 나타내는 (255, 0, 0)을 사용 -> 다각형의 선 두께를 나타내는 정수 값 = 1      
        cv2.polylines(out_img, [pts_right], False, (255,0,0), 1)

        #=====
        # catch: 조향각 반영 속도 조정
        #=====
	    # (0, 400)은 왼쪽 꼭지점의 좌표를, (width, 400)은 오른쪽 꼭지점의 좌표를 나타. 이 다각형은 가로로 놓인 직선으냄로, y 좌표 400에서 평행한 선을 그림
        pts_catch = np.array([[0, 380], [width, 380]], np.int32)
        cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)

        # 337 -> 310
	    # 각 차선의 유효한 픽셀의 인덱스 추출
	    # 왼쪽 차선의 인덱스 저장 - nonzerox가 다음의 조건들을 만족하면 good_left_inds에 저장
        good_left_inds = ((nonzerox >= win_l_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_l_w_r)).nonzero()[0]
	    # 오른쪽 차선의 인덱스 저장 - nonzeroy가 다음의 조건들을 만족하면 good_right_inds에 저장
        good_right_inds = ((nonzerox >= win_r_w_l) & (nonzeroy <= win_h2) & (nonzeroy > win_h1) & (nonzerox <= win_r_w_r)).nonzero()[0]

        line_exist_flag = None # 왼쪽 차선의 존재 여부를 나타냄 -> 왼쪽 차선이 감지되면 True로 설정
        y_current = None # 현재 처리중인 차선의 y좌표
        x_current = None # 현재 처리중인 차선의 x좌표
        good_center_inds = None # 차선 중앙에 해당하는 점들의 인덱스 배열을 저장
        p_cut = None # 차선을 피팅하기 위해 사용되는 다항식을 저장

        # 왼쪽 시작 라인 이전의 최소 점(minipix)를 확인
	    # 왼쪽에 minipix가 있다면 왼쪽을 그리고 왼쪽에 의존하여 오른쪽 그리기
        # 오른쪽에 minipix가 있다면 오른쪽을 그리고 오른쪽에 의존하여 왼쪽 그리기
	    # 오른쪽, 왼쪽 차선의 길이를 비교하여 어느쪽 차선을 우선적으로 추적할지 결정

	    # 오른쪽 차선이 더 긴 경우 - 오른쪽 차선 우선으로 추적
        if len(good_right_inds) > len(good_left_inds):
	    # self.left_cnt + self.right_cnt 값이  TOTAL_CNT(=50)보다 작거나 같을 때 성립. -> 성립한다면 self.right_cnt에 1 더하기 & self.left_cnt에 1 빼기
            if (self.left_cnt + self.right_cnt <= TOTAL_CNT) :
                self.right_cnt += 1
                self.left_cnt -=1
	        # 현재 차선을 "RIGHT" 오른쪽으로 선언
            self.current_line = "RIGHT"
	        #line_flag 값을 2로 설정
            line_flag = 2
	        #np.argmax - 배열에서 최댓값의 인덱스 반환
            #x_current 변수에는 good_right_inds에서 가장 큰 nonzeroy 값을 가진 인덱스에 해당하는 nonzerox 값
            x_current = nonzerox[good_right_inds[np.argmax(nonzeroy[good_right_inds])]]
            # y_current 변수에는 good_right_inds에서 가장 큰 nonzeroy 값을 할당
            y_current = np.int(np.max(nonzeroy[good_right_inds]))
	    # 왼쪽 차선이 더 긴 경우  - 왼쪽 차선 우선으로 추적
        elif len(good_left_inds) > len(good_right_inds):
            # 현재 차선을 "LEFT" 쪽으로 선언
            self.current_line = "LEFT"
	        #line_flag 값을 1로 설정
            line_flag = 1
            # x_current 변수에는 good_left_inds의 nonzerox 값의 평균을 할당
            x_current = np.int(np.mean(nonzerox[good_left_inds]))
            # y_current 변수에는 good_left_inds의 nonzeroy 값의 평균을 할당
            y_current = np.int(np.mean(nonzeroy[good_left_inds]))
	        # max_y 변수에는 y_current 값을 할당
            max_y = y_current


	    # 왼쪽 차선 길이 = 오른쪽 차선 길이
        else:
	        # 현재 차선을 "MID" 중간으로 선언
            self.current_line = "MID"
	        #line_flag 값을 3로 설정
            line_flag = 3
	    # line_flag가 3이 아니라면 = 차선이 중앙이 아닌 왼쪽, 오른쪽을 우선시한다면
        if line_flag != 3:
	        # good_left_inds의 길이만큼 반복하면서 cv2.circle 함수를 이용하여 해당 좌표에 반지름 1의 원을 그림 / 색상은 (255,255,0)으로 설정 / -1은 원의 내부를 채우는 두께 매개변수
            for i in range(len(good_left_inds)):
                    img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (255,255,0), -1)
            # window의 개수만큼 반복
            for window in range(0, nwindows):
		    #왼쪽 차선 우선
                if line_flag == 1: 
                    # 사각형 x,y 범위 설정
                    win_y_low = y_current - (window + 1) * window_height
                    win_y_high = y_current - (window) * window_height
                    win_x_low = x_current - margin
                    win_x_high = x_current + margin
                    # 0.33은 길의 너비
		            # out_img에 사각형 그림 - 사각형의 좌표는 (win_x_low, win_y_low)에서 (win_x_high, win_y_high)로 설정
                    cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 255, 0), 1)
		            # 왼쪽 차선을 시각화 & 현재 창의 좌표에 추가적으로 오른쪽으로 이동한 만큼(width * temp) 그려짐
                    cv2.rectangle(out_img, (win_x_low + int(width * temp), win_y_low), (win_x_high + int(width * temp), win_y_high), (255, 0, 0), 1)
                    # good_left_inds 배열을 업데이트 = 현재 창 내의 점들의 인덱스를 추출하는 과정 -> (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high) 조건을 만족하는 점들의 인덱스를 good_left_inds에 저장
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
		            # 왼쪽 차선이 오른쪽 차선보다 클때 x_current를 다시 설정해줌. -> good_left_inds 배열에 있는 x 좌표들의 평균값을 계산하여  x_current에 다시 할당해줌.
                    if len(good_left_inds) > len(good_right_inds):
                        x_current = np.int(np.mean(nonzerox[good_left_inds]))
                    # nonzeroy[left_lane_inds]와 nonzerox[left_lane_inds]의 값이 비어있지 않을 때
                    elif nonzeroy[left_lane_inds] != [] and nonzerox[left_lane_inds] != []:
			            # 두 개의 배열을 np.ployfit에 전달하여 점들ㅇ르 가장 잘 피팅하는 다항식의 계수 반환하여 p_left에 다항식의 계수 저장
                        p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2) 
			            # 피팅된 다항식에 win_y_high 값을 대입하여 x_current 값을 계산
                        x_current = np.int(np.polyval(p_left, win_y_high))
		            # 338~344의 y좌표 범위에 해당하는 사각형 영역에서 노란색 선 인식 -> 다음 창의 위치 결정
                    if win_y_low >= 338 and win_y_low < 344:
                    	# 0.165는 길의 너비(= 0.33)의 반 길이
                        x_location = x_current + int(width * half_temp) 
                else: # line_flag = 2인 경우 = 오른쪽 차선이 우선인 경우
		            # 현재 창의 위치를 설정하기 위해 win_y_low, win_y_high, win_x_low, win_x_high 값 계산
                    win_y_low = y_current - (window + 1) * window_height
                    win_y_high = y_current - (window) * window_height
                    win_x_low = x_current - margin
                    win_x_high = x_current + margin
		            # cv2.rectangle 함수를 이용하여 현재 창을 사각형으로 시각화 / 범위 : win_x_low - int(width * temp)부터 win_x_high - int(width * temp) = 왼쪽 차선을 고려한 영역을 표시하는 사각형
                    cv2.rectangle(out_img, (win_x_low - int(width * temp), win_y_low), (win_x_high - int(width * temp), win_y_high), (0, 255, 0), 1)
                    cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 1)
		            # 현재 창에 포함된 점들의 인덱스 추출 - 오른쪽 차선에 해당하는 점들
                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
		            # 오른쪽차선이 왼쪽 차선보다 크다면 x_current 값을 계산
                    if len(good_right_inds) > len(good_left_inds):
                        x_current = np.int(np.mean(nonzerox[good_right_inds]))
		            # nonzeroy[right_lane_inds] & nonzerox[right_lane_inds] 값이 비어있지 않을 때
                    elif nonzeroy[right_lane_inds] != [] and nonzerox[right_lane_inds] != []:
			        # 두 개의 배열을 np.ployfit에 전달하여 점들ㅇ르 가장 잘 피팅하는 다항식의 계수 반환하여 p_right에 다항식의 계수 저장
                        p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2) 
			        # 피팅된 다항식에 win_y_high 값을 대입하여 x_current 값을 계산
                        x_current = np.int(np.polyval(p_right, win_y_high))
		            # y 좌표의 범위가 338~344일 때 x_location값을 다시 설정
                    if win_y_low >= 338 and win_y_low < 344:
                        x_location = x_current - int(width * half_temp) 

                left_lane_inds.extend(good_left_inds)
	    # out_img, x_location, self.current_line값을 return 해줌.
        return out_img, x_location, self.current_line


warper = Warper()
slideWindow = SlideWindow()

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse pos (x, y)", x, y)


#=============================================
# 모터 토픽을 발행하는 함수
# 입력으로 받은 angle과 speed 값을
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):

    global motor

    motor_msg = xycar_motor()

    motor_msg.angle = angle
    motor_msg.speed = speed

    motor.publish(motor_msg)

#=============================================
# 실질적인 메인 함수
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함.
#=============================================
def start():

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, cv_image, initialized, flag, last_x_location, x_loc

    ####
    # rospy.init_node('cam_tune', anonymous=True)
    # rospy.init_node('xycar_motor', xycar_motor)
    rospy.init_node('my_driver')

    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

    print ("----- Xycar self driving -----")
    rospy.wait_for_message("/usb_cam/image_raw/", Image)

    while not rospy.is_shutdown():

        # 이미지 처리를 위해 카메라 원본 이미지를 img에 복사 저장한다.
        img = cv_image.copy()


	    # BGR 색상 공간에서 RGB 색상 공간으로 변환 - 이미지의 색상 채널 순서를 조정하는 역할
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	    # 차선의 하한과 상한을 나타내는 RGB값 - 범위에 속하는 픽셀만을 선택 & 나머지는 제거
	    # lower_lane = [minimum_blue, minimum_green, minimum_red]
	    # upper_lane = [maximum_blue, maximum_green, maximum_red]
        lower_lane = np.array([235, 235, 235]) 
        upper_lane = np.array([255, 255, 255])

	    # img에서 lower_lane과 upper_lane 범위에 속하는 픽셀을 선택 - 선택된 픽셀은 흰색으로 표시 / 나머지 픽셀은 검은색으로 표시
        img = cv2.inRange(img, lower_lane, upper_lane)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
        # edge_img = cv2.Canny(np.uint8(blur_gray), 60, 70)

        # cv2.imshow("ProcessedImage", edge_img)
        # cv2.waitKey(1)

        warped_img = warper.warp(img)
        cv2.imshow("original", cv_image)
        cv2.imshow("BEView", warped_img)

        
        cv2.waitKey(1)

        slide_img, x_location, _ = slideWindow.slidewindow(warped_img, x_loc)
        x_loc = x_location

        if x_location == None:
            # flag = True
            x_location = last_x_location
        
        else:
            last_x_location = x_location
            # if flag == True and x_location > 279:
            #     x_location += 13
            #     flag = False
        
        print(x_location)
        cv2.imshow("window_view", slide_img)
        cv2.setMouseCallback("window_view", mouse_callback)
        cv2.waitKey(1)

        angle = 0

        angle = int((279 - int(x_location))*-0.5)
        # speed = max(5, 40 - abs(angle))
        
        speed = 0
        drive(angle, speed)


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함
# start() 함수가 실질적인 메인 함수임.
#=============================================
if __name__ == '__main__':
    start()