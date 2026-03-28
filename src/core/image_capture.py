import cv2
import numpy as np
import os
from datetime import datetime

class ImageCapture:
    def __init__(self):
        self.camera = None
        self.camera_params = {}
        self.frame_count = 0
        self.is_recording = False
        self.video_writer = None
    
    def open_camera(self, camera_id=0, api_preference=cv2.CAP_ANY):
        """
        打开相机
        
        Args:
            camera_id: 相机ID
            api_preference: 后端API偏好
            
        Returns:
            是否成功
        """
        try:
            self.camera = cv2.VideoCapture(camera_id, api_preference)
            if not self.camera.isOpened():
                raise Exception(f"无法打开相机 {camera_id}")
            return True
        except Exception as e:
            print(f"打开相机失败: {e}")
            return False
    
    def close_camera(self):
        """
        关闭相机
        """
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
    
    def set_camera_parameter(self, param_id, value):
        """
        设置相机参数
        
        Args:
            param_id: 参数ID (cv2.CAP_PROP_*)
            value: 参数值
            
        Returns:
            是否成功
        """
        if self.camera is not None:
            success = self.camera.set(param_id, value)
            if success:
                self.camera_params[param_id] = value
            return success
        return False
    
    def get_camera_parameter(self, param_id):
        """
        获取相机参数
        
        Args:
            param_id: 参数ID
            
        Returns:
            参数值
        """
        if self.camera is not None:
            return self.camera.get(param_id)
        return None
    
    def set_resolution(self, width, height):
        """
        设置分辨率
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            是否成功
        """
        if self.camera is not None:
            success1 = self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            success2 = self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return success1 and success2
        return False
    
    def get_resolution(self):
        """
        获取当前分辨率
        
        Returns:
            (width, height)
        """
        if self.camera is not None:
            width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return (int(width), int(height))
        return (0, 0)
    
    def set_fps(self, fps):
        """
        设置帧率
        
        Args:
            fps: 帧率
            
        Returns:
            是否成功
        """
        if self.camera is not None:
            return self.camera.set(cv2.CAP_PROP_FPS, fps)
        return False
    
    def get_fps(self):
        """
        获取当前帧率
        
        Returns:
            帧率
        """
        if self.camera is not None:
            return self.camera.get(cv2.CAP_PROP_FPS)
        return 0
    
    def capture_frame(self):
        """
        捕捉单帧图像
        
        Returns:
            图像帧
        """
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.frame_count += 1
                return frame
        return None
    
    def capture_continuous(self, callback=None, max_frames=None, show_preview=False):
        """
        连续采集图像
        
        Args:
            callback: 每帧回调函数
            max_frames: 最大帧数
            show_preview: 是否显示预览
        """
        if self.camera is None or not self.camera.isOpened():
            print("相机未打开")
            return
        
        frames_captured = 0
        try:
            while True:
                frame = self.capture_frame()
                if frame is None:
                    break
                
                frames_captured += 1
                
                if callback is not None:
                    callback(frame, frames_captured)
                
                if show_preview:
                    cv2.imshow('Camera Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if max_frames is not None and frames_captured >= max_frames:
                    break
        finally:
            if show_preview:
                cv2.destroyWindow('Camera Preview')
    
    def save_image(self, image, directory='images', prefix='capture', format='png'):
        """
        保存图像
        
        Args:
            image: 图像
            directory: 保存目录
            prefix: 文件名前缀
            format: 图像格式
            
        Returns:
            保存的文件路径
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.{format}"
        filepath = os.path.join(directory, filename)
        
        cv2.imwrite(filepath, image)
        return filepath
    
    def start_recording(self, output_file, fps=30, codec='mp4v'):
        """
        开始录制视频
        
        Args:
            output_file: 输出文件路径
            fps: 帧率
            codec: 编码器
            
        Returns:
            是否成功
        """
        if self.camera is None:
            return False
        
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        self.is_recording = True
        return True
    
    def stop_recording(self):
        """
        停止录制视频
        """
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
    
    def record_frame(self, frame):
        """
        录制单帧
        
        Args:
            frame: 图像帧
        """
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
    
    def load_image(self, filepath):
        """
        从文件加载图像
        
        Args:
            filepath: 文件路径
            
        Returns:
            图像
        """
        if os.path.exists(filepath):
            return cv2.imread(filepath)
        return None
    
    def get_available_cameras(self, max_check=10):
        """
        获取可用相机列表
        
        Args:
            max_check: 最大检查数量
            
        Returns:
            可用相机ID列表
        """
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_camera()


class CameraCalibrator:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
    
    def calibrate_from_images(self, image_paths, chessboard_size, square_size):
        """
        从图像标定相机
        
        Args:
            image_paths: 图像路径列表
            chessboard_size: 棋盘格尺寸 (cols, rows)
            square_size: 棋盘格大小 (mm)
            
        Returns:
            是否成功
        """
        obj_points = []
        img_points = []
        
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                obj_points.append(objp)
                
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                img_points.append(corners_refined)
        
        if len(obj_points) < 3:
            return False
        
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        
        return ret
    
    def undistort_image(self, image):
        """
        图像去畸变
        
        Args:
            image: 输入图像
            
        Returns:
            去畸变后的图像
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )
        
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def get_reprojection_error(self, obj_points, img_points):
        """
        计算重投影误差
        
        Args:
            obj_points: 物体点
            img_points: 图像点
            
        Returns:
            平均误差
        """
        if self.camera_matrix is None:
            return None
        
        mean_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(
                obj_points[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
        
        return mean_error / len(obj_points)
