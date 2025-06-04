from rclpy.node import Node
from std_msgs.msg import String
from pros_car_py.car_models import DeviceDataTypeEnum, CarCControl
import threading
import time
import numpy as np
from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
from pros_car_py.detector import group_parallel_lines

class CarController:

    def __init__(self, ros_communicator, nav_processing):
        self.ros_communicator = ros_communicator
        self.nav_processing = nav_processing
        # 用來管理後台執行緒的屬性
        self._auto_nav_thread = None
        self._stop_event = None
        self._thread_running = False
        self.flag = 0

        self._auto_nav_thread = None
        self._stop_event = threading.Event()
        self._thread_running = False

        self.target_idx = 0  # 目標索引
        self.target_list = [
            [0.12577216615733916, 4.207528556910003],
            [0.004709751367064641, -0.43933601070552486],
            [3.202388878639925, 3.893176401328583],
        ]
        self.action_gen = None

    def update_action(self, action_key):
        """
        Updates the velocity for each of the car's wheels.

        Args:
            vel1 (float): Velocity for the rear left wheel (rad/s).
            vel2 (float): Velocity for the rear right wheel (rad/s).
            vel3 (float): Velocity for the front left wheel (rad/s).
            vel4 (float): Velocity for the front right wheel (rad/s).

        Example:
            car_controller.update_velocity(10, 10, 10, 10)  # Set all wheels' velocity to 10 rad/s.
        """
        self.ros_communicator.publish_car_control(action_key)

    def manual_control(self, key):
        """
        Controls the car based on single character inputs ('w', 'a', 's', 'd', 'z').

        Args:
            key (str): A single character representing a control command.
                'w' - move forward
                's' - move backward
                'a' - turn left
                'd' - turn right
                'z' - stop

        Example:
            car_controller.manual_control('w')  # Moves the car forward.
        """
        if key == "w":
            self.update_action("FORWARD")
        elif key == "s":
            self.update_action("BACKWARD")
        elif key == "a":
            self.update_action("LEFT_FRONT")
        elif key == "d":
            self.update_action("RIGHT_FRONT")
        elif key == "e":
            self.update_action("COUNTERCLOCKWISE_ROTATION")
        elif key == "r":
            self.update_action("CLOCKWISE_ROTATION")
        elif key == "z":
            self.update_action("STOP")
        elif key == "q":
            self.update_action("STOP")
            time.sleep(0.1)
            return True
        else:
            pass
        wall_edges = self.ros_communicator.get_wall_edges()
        door_edges = self.ros_communicator.get_door_edges()
        pole_edges = self.ros_communicator.get_pole_edges()
        # door_groups = group_parallel_lines(door_edges, spatial_eps=20, angle_eps=np.radians(10), min_samples=2)
        cnt_wall_edges = len(wall_edges)//4 if wall_edges else 0
        cnt_door_edges = len(door_edges)//4 if door_edges else 0
        if cnt_wall_edges and cnt_door_edges:
            print(self.nav_processing.check_end_close(door_edges, wall_edges))
        # print(door_groups)
        # print(f"[Door Check] Found {len(door_groups)} door groups.")
        # all_groups = {
        #     "door_edge":door_groups
        # }
        cnt_door_edges = cnt_wall_edges = cnt_pole_edges = 0
        # if wall_edges:
        #     cnt_wall_edges = len(wall_edges) / 4
        #     print(f'wall edge count: {cnt_wall_edges}')
        #     # print(f'wall edges: {wall_edges}')
        # if pole_edges:
        #     cnt_pole_edges = len(pole_edges) / 4
        #     print(f'pole edge count: {cnt_pole_edges}')
        #     print(f'pole edges: {pole_edges}')
        # if door_edges:
        #     cnt_door_edges = len(door_edges) / 4
        #     print(f'door edge count: {cnt_door_edges}')
        #     # print(f'door edges: {door_edges}')
        print('')

    def auto_control(self, mode="manual_auto_nav", target=None, key=None):
        """
        自動控制邏輯
        Args:
            mode: 控制模式 ("auto_nav" 或 "manual_nav")
            target: 目標座標 (用於 manual_nav 模式)
            key: 鍵盤輸入
        """
        # 如果有按鍵輸入
        if self.flag == 0:
            stop_event = threading.Event()
            thread = threading.Thread(target=self.background_task, args=(stop_event,))

        if key == "q":
            # 按下 q 時停止導航並退出
            if self._thread_running:
                self._stop_event.set()
                self._auto_nav_thread.join()
                self._thread_running = False

            self.nav_processing.reset_nav_process()
            action_key = "STOP"
            self.ros_communicator.publish_car_control(
                action_key, publish_rear=True, publish_front=True
            )
            return True

        if not self._thread_running:
            self._stop_event.clear()  # 清除之前的停止狀態
            self.action_gen = None
            self.nav_processing.level_state = 0
            self._auto_nav_thread = threading.Thread(
                target=self.background_task,
                args=(self._stop_event, mode, target),
                daemon=True,
            )
            self.nav_processing.rst_flag = False
            self.nav_processing.left_end_close = False
            self._auto_nav_thread.start()
            self._thread_running = True

        return False

    def stop_nav(self):
        for i in range(20):
            time.sleep(0.1)
            self.update_action("STOP")

    def background_task(self, stop_event, mode, target):
        """
        後台任務：不斷執行導航動作直到 stop_event 被設定。
        """
        if mode == 'random_living_room_nav':
            pose_metadata = self.ros_communicator.get_aruco_estimated_car_pose()
            position = pose_metadata.position
            orientation = pose_metadata.orientation
            cur_yaw = get_yaw_from_quaternion(orientation.z, orientation.w)
            # self.nav_processing.reset_start(position, cur_yaw)
        while not stop_event.is_set():
            # self.nav_processing.cnt += 1
            if mode == "manual_auto_nav":
                action_key = (
                    self.nav_processing.get_action_from_nav2_plan_no_dynamic_p_2_p(
                        goal_coordinates=None
                    )
                )
                if self.nav_processing.get_finish_flag():
                    self.nav_processing.reset_nav_process()
            elif mode == "target_auto_nav":

                current_target = self.target_list[self.target_idx]
                action_key = (
                    self.nav_processing.get_action_from_nav2_plan_no_dynamic_p_2_p(
                        goal_coordinates=current_target
                    )
                )
                if self.nav_processing.get_finish_flag():
                    self.nav_processing.reset_nav_process()
                    self.target_idx = (self.target_idx + 1) % len(self.target_list)
                    continue
            # 發布控制指令

            elif mode == "custom_nav":
                action_key = self.nav_processing.camera_nav_unity()
                
            elif mode == "fix_living_room_nav":
                action_key = self.nav_processing.fix_living_room_nav()
                
            elif mode == 'random_living_room_nav':
                # print('car pose:', self.ros_communicator.get_aruco_estimated_car_pose())
                detection_status = self.ros_communicator.get_latest_yolo_detection_status().data
                # action_key = self.nav_processing.random_living_room_nav(position, cur_yaw, detection_status)
                # self.action_gen = self.nav_processing.random_living_room_nav(position, cur_yaw, detection_status)
                action_key = self.nav_processing.random_living_room_nav(position, cur_yaw, detection_status)
                # try:
                #     action_key = next(self.action_gen)
                # except StopIteration:
                #     self.action_gen = None
                #     print('stop!!!!!!!!!!!!!!')
                #     action_key = "STOP"
                
            elif mode == 'random_door_nav':
                if self.action_gen:
                    try:
                        action_key = next(self.action_gen)
                    except StopIteration:
                        action_key = "STOP"
                        self.action_gen = None
                else:
                    self.nav_processing.reset_nav_process()
                    self.action_gen = self.nav_processing.random_door_nav()
                    try:
                        action_key = next(self.action_gen)
                    except StopIteration:
                        self.action_gen = None
                        action_key = "STOP"
                
            if self._thread_running == False:
                action_key = "STOP"
                
            print(action_key)
            time.sleep(0.05)
            self.ros_communicator.publish_car_control(
                action_key, publish_rear=True, publish_front=True
            )
        
        # 收尾動作
        print("[background_task] Navigation stopped.")

    def run(self, mode, target):
        pass
