from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math
from pros_car_py.path_planning import PlannerRRTStar, MapLoader
import os
import numpy as np
import random
import time
from pros_car_py.detector import group_parallel_lines

class Nav2Processing:
    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False
        self.K = np.array([
            [576.83946  , 0.0       , 319.59192 ],
            [0.         , 577.82786 , 238.89255 ],
            [0.         , 0.        , 1.        ]
        ])
        self.rst_flag = False
        self.level = 1
        self.left_end_close = False
        self.right_end_close = False
        self.level1_backward =False
        self.level_state = 0;
        self.cnt = 0
        self.rotate_flag = True
        
    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        # 只抓第一次路径
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"
            else:
                print("Get first path")
                self.index = 0
                self.global_plan_msg = (
                    self.data_processor.get_processed_received_global_plan_no_dynamic()
                )
                self.recordFlag = 1
                action_key = "STOP"

        car_position, car_orientation = self.data_processor.get_processed_amcl_pose()

        goal_position = self.ros_communicator.get_latest_goal()
        target_distance = cal_distance(car_position, goal_position)

        # 抓最近的物標(可調距離)
        target_x, target_y = self.get_next_target_point(car_position)

        if target_x is None or target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 計算角度誤差
        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def check_data_availability(self):
        return (
            self.data_processor.get_processed_received_global_plan_no_dynamic()
            and self.data_processor.get_processed_amcl_pose()
            and self.ros_communicator.get_latest_goal()
        )

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]

    def camera_nav(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        if camera_multi_depth == None or yolo_target_info == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])

        action = "STOP"
        limit_distance = 0.7

        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 0.8:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action

    def camera_nav_unity(self):
        """
        YOLO 目標資訊 (yolo_target_info) 說明：

        - 索引 0 (index 0)：
            - 表示是否成功偵測到目標
            - 0：未偵測到目標
            - 1：成功偵測到目標

        - 索引 1 (index 1)：
            - 目標的深度距離 (與相機的距離，單位為公尺)，如果沒偵測到目標就回傳 0
            - 與目標過近時(大約 40 公分以內)會回傳 -1

        - 索引 2 (index 2)：
            - 目標相對於畫面正中心的像素偏移量
            - 若目標位於畫面中心右側，數值為正
            - 若目標位於畫面中心左側，數值為負
            - 若沒有目標則回傳 0

        畫面 n 個等分點深度 (camera_multi_depth) 說明 :

        - 儲存相機畫面中央高度上 n 個等距水平點的深度值。
        - 若距離過遠、過近（小於 40 公分）或是實體相機有時候深度會出一些問題，則該點的深度值將設定為 -1。
        """
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()
        yolo_target_info[1] *= 100.0
        camera_multi_depth = list(
            map(lambda x: x * 100.0, self.data_processor.get_camera_x_multi_depth())
        )

        if camera_multi_depth == None or yolo_target_info[0] == None:
            return "STOP"

        camera_forward_depth = self.filter_negative_one(camera_multi_depth[7:13])
        camera_left_depth = self.filter_negative_one(camera_multi_depth[0:7])
        camera_right_depth = self.filter_negative_one(camera_multi_depth[13:20])
        action = "STOP"
        limit_distance = 10.0
        # print(yolo_target_info[1])
        print(yolo_target_info)
        if all(depth > limit_distance for depth in camera_forward_depth):
            if yolo_target_info[0] == 1:
                if yolo_target_info[2] > 200.0:
                    action = "CLOCKWISE_ROTATION_SLOW"
                elif yolo_target_info[2] < -200.0:
                    action = "COUNTERCLOCKWISE_ROTATION_SLOW"
                else:
                    if yolo_target_info[1] < 2.0:
                        action = "STOP"
                    else:
                        action = "FORWARD_SLOW"
            else:
                action = "FORWARD"
        elif any(depth < limit_distance for depth in camera_left_depth):
            action = "CLOCKWISE_ROTATION"
        elif any(depth < limit_distance for depth in camera_right_depth):
            action = "COUNTERCLOCKWISE_ROTATION"
        return action
    
    def fix_living_room_nav(self):
        detection_status = self.ros_communicator.get_latest_yolo_detection_status().data
        
        # print(detection_status)
        if detection_status:
            detection_bbox = self.ros_communicator.get_latest_yolo_target_info().data
            img_width = self.ros_communicator.latest_map.info.width
            img_width = 640
            center_x_min = int(img_width * 0.4)
            center_x_max = int(img_width * 0.6)
            x_min, y_min, x_max, y_max = detection_bbox
            bbox_center_x = (x_min + x_max) // 2
            if bbox_center_x < center_x_min:
                return "COUNTERCLOCKWISE_ROTATION"  # Pikachu is on the left → turn left
            elif bbox_center_x > center_x_max:
                return "CLOCKWISE_ROTATION"  # Pikachu is on the right → turn right
            else:
                return "FORWARD"
            
        return "COUNTERCLOCKWISE_ROTATION"
    def position_to_pixel(self, position):
        """
        position: (x, y) in meters
        return: (u, v) in pixels
        """
        x, y = position
        u = int((x - self.ros_communicator.latest_map.info.origin.position.x) / self.ros_communicator.latest_map.info.resolution)
        v = int((y - self.ros_communicator.latest_map.info.origin.position.y) / self.ros_communicator.latest_map.info.resolution)

        # v = self.height - v
        return (u, v)
    
    def check_horizontal_middle(self, edge):
        x1, y1 = edge[0], edge[1]
        x2, y2 = edge[2], edge[3]
        horizontal_tolerance = 2           # pixels
        middle_band_width = 60             # horizontal "middle" band, e.g., ±30 pixels around center

        # is_horizontal = abs(y1 - y2) <= horizontal_tolerance
        # middle_x = (x1 + x2) / 2
        # is_in_middle = abs(middle_x - 320) <= (middle_band_width / 2)
        for i, _ in enumerate(edge[::4]):
            x1, y1 = edge[i], edge[i+1]
            x2, y2 = edge[i+2], edge[i+3]
            is_horizontal = abs(y1 - y2) <= horizontal_tolerance
            is_in_middle = 100 <= abs(x1 - x2) <= 180
            if is_horizontal and is_in_middle:
                return True
        return False
    
    def check_hole(self, edge):
        for x, y in edge[::2]:
            if x == 0.0:
                return True
        return False
    
    def check_end_close(self, door_edges, wall_edges):
        wall_x1, wall_y1 = wall_edges[0], wall_edges[1]
        wall_x2, wall_y2 = wall_edges[2], wall_edges[3]
        for i in range(0, len(door_edges), 2):
            x, y = door_edges[i:i+2]
            if abs(wall_x1 - x) <= 5 and abs(wall_y1 - y) <= 5:
                return True
            if abs(wall_x2 - x) <= 5 and abs(wall_y2 - y) <= 5:
                return True
        return False
    
    def reset_start(self, position, cur_yaw, detection_status):
        target_pos = (1.8, 1.6)
        dx, dy = target_pos[0] - position.x, target_pos[1] - position.y
        distance = np.hypot(dx, dy)
        desired_yaw = np.rad2deg(math.atan2(dy, dx))
        angle_diff = desired_yaw - cur_yaw
        
        yaw_tolerance = 10.0  # degrees
        distance_tolerance = 0.4  # meters
        
        if detection_status:
            self.rst_flag = True
        if abs(angle_diff) > yaw_tolerance:
            if angle_diff > 0:
                return "COUNTERCLOCKWISE_ROTATION"
            else:
                return "CLOCKWISE_ROTATION"
            
        elif distance > distance_tolerance:
            return "FORWARD"
        else:
            self.rst_flag = True
            return "STOP"
        
    def random_living_room_nav(self, position, cur_yaw, detection_status):
        door_edges = self.ros_communicator.get_door_edges()
        cnt_door_edges = len(door_edges) // 4 if door_edges else 0
        
        if detection_status:
            print('Found Pikachu!!!')
            detection_bbox = self.ros_communicator.get_latest_yolo_target_info().data
            img_width = 640  # or get from self.ros_communicator.latest_map.info.width
            center_x_min = int(img_width * 0.35)
            center_x_max = int(img_width * 0.65)
            x_min, y_min, x_max, y_max = detection_bbox
            bbox_center_x = (x_min + x_max) // 2

            if bbox_center_x < center_x_min:
                return "COUNTERCLOCKWISE_ROTATION"  # Pikachu is on the left → turn left
            elif bbox_center_x > center_x_max:
                return "CLOCKWISE_ROTATION"  # Pikachu is on the right → turn right
            elif y_max >= 479.0:
                return "STOP"
            else:
                return "FORWARD"  # Pikachu is in front → move forward
        else:
            print(self.level_state)
            door_edges = self.ros_communicator.get_door_edges()
            cnt_door_edges = len(door_edges) // 4 if door_edges else 0
            pole_edges = self.ros_communicator.get_pole_edges()
            cnt_pole_edges = len(pole_edges) // 4 if pole_edges else 0
            match self.level_state:
                case 0:
                    if cnt_door_edges == 2 and cnt_pole_edges == 3:
                        self.level_state = 1
                    return "COUNTERCLOCKWISE_ROTATION"
                case 1: 
                    if (door_edges[1] == 479.0 or door_edges[3] == 479) and cnt_door_edges == 1:
                        self.level_state = 2
                    return "CLOCKWISE_ROTATION"
                case 2:
                    if cnt_pole_edges == 1 and door_edges[1] == 479.0 and door_edges[3] == 479:
                        self.level_state = 3
                    return "FORWARD"
                case 3:
                    # if detection_status:
                    #     self.level_state = 4
                    time.sleep(0.8)
                    return "COUNTERCLOCKWISE_ROTATION"
                case 4:
                    return "FORWARD"
        return "STOP"
        
    def random_door_nav(self):
        # print(self.ros_communicator.latest_aruco_marker_list)
        detected_markers = self.ros_communicator.latest_aruco_marker_list
        show_state = True
        print('Start initial position reset')
        while True:
            pole_edges = self.ros_communicator.get_pole_edges()
            pt1 = pole_edges[1]
            pt2 = pole_edges[3]
            if pt1 < 450 and pt2 < 450:
                if show_state:print("[Step 1] Floor detected, stop reversing.")
                break
            if show_state:print("[Step 1] Still no floor, continue reversing.")
            yield 'BACKWARD'
        yield "STOP"
        
        turn_side = 1
        if show_state:print("[Step 2] Start scanning for walls and doors...")
        while True:
            leave_wall = False
            if show_state:print("[Wall Search] Rotating to find wall...")
            while True:
                yield "COUNTERCLOCKWISE_ROTATION" if turn_side else "CLOCKWISE_ROTATION"
                wall_edges = self.ros_communicator.get_wall_edges()
                door_edges = self.ros_communicator.get_door_edges()
                pole_edges = self.ros_communicator.get_pole_edges()
                cnt_wall_edges = len(wall_edges)//4 if wall_edges else 0
                cnt_door_edges = len(door_edges)//4 if door_edges else 0
                if cnt_wall_edges and cnt_door_edges:
                    if self.check_end_close(door_edges, wall_edges):
                        turn_side *= -1
                        
                # if wall_edges:
                #     if show_state:print("[Wall Search] Wall detected!")
                #     is_detected = self.ros_communicator.get_latest_yolo_detection_status().data
                    
                #     if is_detected:
                #         if show_state:
                #             print("[Wall Search] Pikachu detected, aligning temporarily...")
                #             detection_bbox = self.ros_communicator.get_latest_yolo_target_info().data
                #             img_width = 640  # or get from self.ros_communicator.latest_map.info.width
                #             center_x_min = int(img_width * 0.3)
                #             center_x_max = int(img_width * 0.7)
                #             x_min, y_min, x_max, y_max = detection_bbox
                #             bbox_center_x = (x_min + x_max) // 2

                #             if bbox_center_x < center_x_min:
                #                 yield "COUNTERCLOCKWISE_ROTATION"  # Pikachu is on the left → turn left
                #             elif bbox_center_x > center_x_max:
                #                 yield "CLOCKWISE_ROTATION"  # Pikachu is on the right → turn right
                #             else:
                #                 yield "FORWARD"
                #         if not leave_wall:
                #             print("[Wall Search] Still hugging wall, keep rotating.")
                #             continue
                #         if show_state:print("[Wall Search] Left wall and found again, stop rotating.")
                #         yield "STOP"
                #         break
                #     else:
                #         leave_wall = True
        
            door_edges = self.ros_communicator.get_door_edges()
            door_groups = group_parallel_lines(door_edges, spatial_eps=20, angle_eps=np.radians(10), min_samples=3)
            if show_state:print(f"[Door Check] Found {len(door_groups)} door groups.")
            all_groups = {
                "door_edge":door_groups
            }
        # if self.level == 1:
        #     print('STATE:', self.level_state)
        #     if not self.level1_backward:
        #         pt1 = pole_edges[1]
        #         pt2 = pole_edges[3]
        #         if pt1 < 450 and pt2 < 450:
        #             self.level1_backward = True
        #             return "STOP"
        #         return 'BACKWARD'
        #     else:
        #         if detected_markers == [0] and cnt_wall_edges == 1 and cnt_door_edges == 2 and cnt_pole_edges == 1:
        #             self.left_end_close = True
        #         if self.level_state == 0:
        #             if cnt_wall_edges == 1 and cnt_door_edges == 0 and cnt_pole_edges == 1:
        #                 self.level_state = 1
        #             if ( 3 in detected_markers and cnt_pole_edges == 3):
        #                 self.level_state = 3
        #                 self.door_pass = 1
        #             if (5 in detected_markers and cnt_pole_edges == 3):
        #                 self.level_state = 3
        #                 self.door_pass = 2
        #             return "COUNTERCLOCKWISE_ROTATION"
        #         elif self.level_state == 1:
        #             if wall_edges[1] == 479.0 and wall_edges[3] == 479.0:
        #                 self.level_state = 2
        #             return "FORWARD"
        #         elif self.level_state == 2:
        #             if cnt_pole_edges >= 1 and cnt_wall_edges == 1 and cnt_door_edges == 1 and not self.left_end_close:
        #                 self.door_pass = 0
        #                 self.level_state = 3
        #             elif self.left_end_close:
        #                 self.door_pass = 3
        #                 self.level_state = 5
        #             return "CLOCKWISE_ROTATION"
        #         elif self.level_state == 3:
        #             for i,_ in enumerate(door_edges[::4]):
        #                 y1, y2 = door_edges[i+1], door_edges[i+3]
        #                 if y1 == 479.0 and y2 == 479.0:
        #                     self.level_state = 4
        #             # if door_edges[1] == 479.0 and door_edges[3] == 479.0:
        #             #     self.level_state = 4
        #             return "FORWARD"
        #         elif self.level_state == 5:
        #             if cnt_pole_edges == 1 and cnt_door_edges == 1:
        #                 self.level_state = 3
        #             return "COUNTERCLOCKWISE_ROTATION"
        #         else:
        #             self.level = 2
        #             self.level_state = 0
        #             self.rst_flag = False
        #             self.rotate_flag = 0 if self.door_pass in [0, 1] else 1
        #             return "STOP"
        
            
        return "STOP"

    def stop_nav(self):
        return "STOP"
