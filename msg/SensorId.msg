# sensor_id enum:
uint8 INVALID_SENSOR_ID = 0
# camera 1~19
uint8 CAM_FRONT_NARROW = 1 # front narrow pinhole
uint8 CAM_FRONT_WIDE = 2 # front wide pinhole
uint8 CAM_LEFT_FRONT = 3 # left toward front pinhole
uint8 CAM_LEFT_BACK = 4 # left toward backward pinhole
uint8 CAM_RIGHT_BACK = 5 # right toward backward pinhole
uint8 CAM_RIGHT_FRONT = 6 # right toward front pinhole
uint8 CAM_BACK = 7 # back middle pinhole
uint8 CAM_FISHEYE_FRONT = 10 # fornt middle fisheye
uint8 CAM_FISHEYE_LEFT = 11 # left middle fisheye
uint8 CAM_FISHEYE_RIGHT = 12 # right middle fisheye
uint8 CAM_FISHEYE_BACK = 13 # back middle fisheye

# lidar,M1,E1*2, 20~29
uint8  LIDAR_RUBY_TOP = 20
uint8  LIDAR_M1_FRONT = 21
uint8  LIDAR_M1_LEFT = 22
uint8  LIDAR_M1_RIGHT = 23
uint8  LIDAR_M1_BACK = 24
uint8  LIDAR_E1_LEFT = 25
uint8  LIDAR_E1_RIGHT = 26

# radar,MRR,SRR*2,30~39
uint8  RADAR_FRONT_MRR = 30
uint8  RADAR_LEFT_BACK_SRR = 31
uint8  RADAR_RIGHT_BACK_SRR = 32
