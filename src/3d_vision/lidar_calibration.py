"""
激光雷达标定模块
包含相机-激光雷达外参标定和激光雷达内参标定
"""

import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.optimize import minimize, least_squares
import json
import os


class LiDARCalibration:
    """激光雷达标定类"""
    
    def __init__(self):
        self.extrinsic_matrix = np.eye(4)  # 外参矩阵
        self.intrinsic_matrix = None       # 相机内参
        self.lidar_to_camera = None        # 激光雷达到相机的变换
        self.camera_to_lidar = None        # 相机的激光雷达的变换
        self.calibration_error = None      # 标定误差
        
    # ==================== 相机-激光雷达外参标定 ====================
    
    def calibrate_extrinsic(self, image_points_2d, lidar_points_3d, 
                           camera_intrinsic, dist_coeffs=None):
        """
        相机-激光雷达外参标定（基于PnP + ICP优化）
        
        Args:
            image_points_2d: 图像2D点 (N, 2) 像素坐标
            lidar_points_3d: 对应的3D点 (N, 3) 激光雷达坐标
            camera_intrinsic: 相机内参矩阵 (3, 3)
            dist_coeffs: 畸变系数
            
        Returns:
            外参矩阵 (4, 4) - 激光雷达到相机的变换
        """
        if len(image_points_2d) < 4 or len(lidar_points_3d) < 4:
            raise ValueError("需要至少4个对应点进行标定")
        
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)
        
        self.intrinsic_matrix = camera_intrinsic
        
        # 步骤1: 使用PnP求解初始外参
        object_points = lidar_points_3d.astype(np.float64)
        image_points = image_points_2d.astype(np.float64)
        
        # 求解PnP问题
        success, rvec, tvec = cv2.solvePnP(
            object_points, 
            image_points, 
            camera_intrinsic, 
            dist_coeffs
        )
        
        if not success:
            raise ValueError("PnP求解失败")
        
        # 转换为旋转矩阵
        R = cv2.Rodrigues(rvec)[0]
        
        # 构建初始外参矩阵（ lidar -> camera）
        self.lidar_to_camera = np.eye(4)
        self.lidar_to_camera[:3, :3] = R
        self.lidar_to_camera[:3, 3] = tvec.flatten()
        
        # 计算逆变换（camera -> lidar）
        self.camera_to_lidar = np.linalg.inv(self.lidar_to_camera)
        
        # 步骤2: 使用ICP优化外参
        self._optimize_extrinsic_icp(image_points_2d, lidar_points_3d, 
                                     camera_intrinsic, dist_coeffs)
        
        return self.lidar_to_camera
    
    def _optimize_extrinsic_icp(self, image_points_2d, lidar_points_3d,
                                camera_intrinsic, dist_coeffs, max_iterations=50):
        """
        使用ICP优化外参
        
        Args:
            image_points_2d: 图像2D点
            lidar_points_3d: 3D点
            camera_intrinsic: 相机内参
            dist_coeffs: 畸变系数
            max_iterations: 最大迭代次数
        """
        # 初始化变换参数 [rx, ry, rz, tx, ty, tz]
        R = self.lidar_to_camera[:3, :3]
        t = self.lidar_to_camera[:3, 3]
        rvec = cv2.Rodrigues(R)[0].flatten()
        
        params = np.concatenate([rvec, t])
        
        # 定义目标函数
        def objective(params):
            rvec = params[:3]
            t = params[3:]
            
            # 构建变换矩阵
            R_new = cv2.Rodrigues(rvec)[0]
            T = np.eye(4)
            T[:3, :3] = R_new
            T[:3, 3] = t
            
            # 投影3D点到图像
            projected, _ = cv2.projectPoints(
                lidar_points_3d, rvec, t, camera_intrinsic, dist_coeffs
            )
            projected = projected.reshape(-1, 2)
            
            # 计算重投影误差
            errors = np.linalg.norm(projected - image_points_2d, axis=1)
            return np.sum(errors ** 2)
        
        # 优化
        result = minimize(objective, params, method='Levenberg-Marquardt')
        
        # 更新外参
        rvec_opt = result.x[:3]
        t_opt = result.x[3:]
        
        R_opt = cv2.Rodrigues(rvec_opt)[0]
        self.lidar_to_camera[:3, :3] = R_opt
        self.lidar_to_camera[:3, 3] = t_opt
        self.camera_to_lidar = np.linalg.inv(self.lidar_to_camera)
        
        # 计算标定误差
        self.calibration_error = np.sqrt(result.fun / len(image_points_2d))
        
    def calibrate_with_chessboard(self, lidar_points, image, 
                                  camera_intrinsic, board_size=(9, 6),
                                  square_size=0.03):
        """
        使用棋盘格进行自动标定
        
        Args:
            lidar_points: 激光雷达点云 (N, 3)
            image: 相机图像
            camera_intrinsic: 相机内参
            board_size: 棋盘格尺寸 (宽, 高)
            square_size: 棋盘格格子大小（米）
            
        Returns:
            外参矩阵
        """
        # 检测棋盘格角点
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        ret, corners = cv2.findChessboardCorners(gray, board_size)
        
        if not ret:
            raise ValueError("未检测到棋盘格")
        
        # 亚像素级角点细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 生成3D棋盘格角点
        obj_points = []
        for i in range(board_size[1]):
            for j in range(board_size[0]):
                obj_points.append([j * square_size, i * square_size, 0])
        obj_points = np.array(obj_points, dtype=np.float32)
        
        # 估计棋盘格平面在激光雷达坐标系中的位置
        # 通过距离图像中检测到的平面区域来筛选激光雷达点
        image_corners = corners_sub.reshape(-1, 2)
        
        # 使用平面分割方法找地面/棋盘格平面
        lidar_plane, inliers = self._find_lidar_plane(lidar_points)
        
        if lidar_plane is None:
            raise ValueError("无法从点云中找到平面")
        
        # 估算棋盘格位置
        lidar_corners_3d = self._estimate_chessboard_lidar_corners(
            lidar_points, image_corners, camera_intrinsic, 
            self.lidar_to_camera if self.lidar_to_camera is not None else np.eye(4)
        )
        
        if len(lidar_corners_3d) < 4:
            raise ValueError("无法估算棋盘格3D位置")
        
        # 标定外参
        self.calibrate_extrinsic(image_corners, lidar_corners_3d, camera_intrinsic)
        
        return self.lidar_to_camera
    
    def _find_lidar_plane(self, points, distance_threshold=0.02, max_iterations=1000):
        """
        使用RANSAC从点云中分割平面
        
        Args:
            points: 点云 (N, 3)
            distance_threshold: 距离阈值
            max_iterations: 最大迭代次数
            
        Returns:
            平面参数 (a, b, c, d) 和内点索引
        """
        best_plane = None
        best_inliers = []
        
        for _ in range(max_iterations):
            # 随机采样3个点
            indices = np.random.choice(len(points), 3, replace=False)
            sample = points[indices]
            
            # 计算平面法向量
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
                
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample[0])
            
            # 计算所有点到平面的距离
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.where(distances < distance_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = np.concatenate([normal, [d]])
        
        return best_plane, np.array(best_inliers)
    
    def _estimate_chessboard_lidar_corners(self, lidar_points, image_corners,
                                           camera_intrinsic, transform):
        """
        估算棋盘格角点的3D位置
        
        Args:
            lidar_points: 激光雷达点云
            image_corners: 图像角点
            camera_intrinsic: 相机内参
            transform: 激光雷达到相机的变换
            
        Returns:
            3D角点坐标
        """
        # 构建KD树用于最近邻搜索
        tree = KDTree(lidar_points)
        
        # 对每个图像角点，找到最近的激光雷达点
        lidar_corners = []
        
        for corner in image_corners:
            # 将图像点投影到归一化平面
            x = (corner[0] - camera_intrinsic[0, 2]) / camera_intrinsic[0, 0]
            y = (corner[1] - camera_intrinsic[1, 2]) / camera_intrinsic[1, 1]
            
            # 使用已知的变换估算深度
            # 这里简化处理：假设棋盘格在某个深度范围内
            depths = np.linspace(0.5, 5.0, 100)
            
            best_point = None
            min_distance = float('inf')
            
            for depth in depths:
                # 假设点在相机坐标系前方
                point_cam = np.array([x * depth, y * depth, depth])
                
                # 转换到激光雷达坐标系
                point_lidar = self._transform_point(point_cam, transform)
                
                # 找最近的激光雷达点
                dist, idx = tree.query(point_lidar)
                
                if dist < min_distance:
                    min_distance = dist
                    best_point = point_lidar
            
            if best_point is not None:
                lidar_corners.append(best_point)
        
        return np.array(lidar_corners)
    
    def _transform_point(self, point, transform):
        """应用4x4变换矩阵到3D点"""
        point_h = np.append(point, 1)
        transformed = np.dot(transform, point_h)
        return transformed[:3]
    
    # ==================== 激光雷达内参标定 ====================
    
    def calibrate_lidar_intrinsic(self, measurements, reference_distance=None):
        """
        激光雷达内参标定
        
        Args:
            measurements: 距离测量值列表 (N,)
            reference_distance: 参考距离（真值）
            
        Returns:
            内参校正因子
        """
        if reference_distance is None:
            # 使用多次测量的平均值作为参考
            reference_distance = np.median(measurements)
        
        # 距离误差模型: measured = k * actual + b
        # 简化为线性校正
        
        # 计算校正因子
        measured_mean = np.mean(measurements)
        scale_factor = reference_distance / measured_mean if measured_mean > 0 else 1.0
        
        # 存储内参
        self.intrinsic_params = {
            'scale': scale_factor,
            'offset': 0,
            'reference_distance': reference_distance,
            'measured_mean': measured_mean
        }
        
        return self.intrinsic_params
    
    def calibrate_lidar_angle_errors(self, measurements, angles, reference_angles=None):
        """
        激光雷达角度误差标定
        
        Args:
            measurements: 测量角度 (N,)
            angles: 理论角度 (N,) 或 标称角度数组
            reference_angles: 参考角度（如果有）
            
        Returns:
            角度校正偏移
        """
        if reference_angles is None:
            reference_angles = angles
        
        # 计算角度偏移
        angle_offsets = reference_angles - measurements
        
        # 去除异常值后取平均
        median_offset = np.median(angle_offsets)
        
        self.angle_intrinsic = {
            'offset': median_offset,
            'measurement_count': len(measurements)
        }
        
        return self.angle_intrinsic
    
    def correct_lidar_measurement(self, distance, angle_h, angle_v=None):
        """
        校正激光雷达测量值
        
        Args:
            distance: 原始距离测量
            angle_h: 水平角度
            angle_v: 垂直角度（可选）
            
        Returns:
            校正后的测量值
        """
        corrected = distance
        
        if hasattr(self, 'intrinsic_params'):
            # 应用距离校正
            corrected = distance * self.intrinsic_params['scale'] + \
                       self.intrinsic_params.get('offset', 0)
        
        if hasattr(self, 'angle_intrinsic'):
            # 应用角度校正
            angle_h = angle_h + self.angle_intrinsic.get('offset', 0)
        
        return corrected, angle_h, angle_v
    
    # ==================== 点云投影 ====================
    
    def project_lidar_to_image(self, lidar_points, image_shape=None):
        """
        将激光雷达点投影到图像
        
        Args:
            lidar_points: 激光雷达点 (N, 3)
            image_shape: 图像尺寸 (H, W)
            
        Returns:
            投影到图像平面的点 (N, 2) 和深度 (N,)
        """
        if self.lidar_to_camera is None:
            raise ValueError("请先进行外参标定")
        
        if self.intrinsic_matrix is None:
            raise ValueError("请提供相机内参")
        
        # 转换到相机坐标系
        points_h = np.hstack([lidar_points, np.ones((len(lidar_points), 1))])
        points_cam = (self.lidar_to_camera @ points_h.T).T
        
        # 过滤在相机后方的点
        valid = points_cam[:, 2] > 0
        
        # 投影到图像平面
        points_2d = np.zeros((len(lidar_points), 2))
        depths = np.zeros(len(lidar_points))
        
        valid_points = points_cam[valid]
        x = valid_points[:, 0] / valid_points[:, 2]
        y = valid_points[:, 1] / valid_points[:, 2]
        
        # 应用相机内参
        u = self.intrinsic_matrix[0, 0] * x + self.intrinsic_matrix[0, 2]
        v = self.intrinsic_matrix[1, 1] * y + self.intrinsic_matrix[1, 2]
        
        points_2d[valid, 0] = u
        points_2d[valid, 1] = v
        depths[valid] = valid_points[:, 2]
        
        # 如果提供了图像形状，则过滤图像外的点
        if image_shape is not None:
            h, w = image_shape[:2]
            in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                      (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h) & valid
        else:
            in_image = valid
        
        return points_2d[in_image], depths[in_image]
    
    def transform_point_cloud(self, points, direction='lidar_to_camera'):
        """
        变换点云坐标系
        
        Args:
            points: 点云 (N, 3)
            direction: 'lidar_to_camera' 或 'camera_to_lidar'
            
        Returns:
            变换后的点云
        """
        if direction == 'lidar_to_camera':
            if self.lidar_to_camera is None:
                raise ValueError("请先进行外参标定")
            transform = self.lidar_to_camera
        else:
            if self.camera_to_lidar is None:
                raise ValueError("请先进行外参标定")
            transform = self.camera_to_lidar
        
        points_h = np.hstack([points, np.ones((len(points), 1))])
        transformed = (transform @ points_h.T).T
        
        return transformed[:, :3]
    
    # ==================== 保存和加载 ====================
    
    def save_calibration(self, filename):
        """
        保存标定参数到文件
        
        Args:
            filename: 保存路径
        """
        data = {
            'lidar_to_camera': self.lidar_to_camera.tolist() if self.lidar_to_camera is not None else None,
            'camera_to_lidar': self.camera_to_lidar.tolist() if self.camera_to_lidar is not None else None,
            'camera_intrinsic': self.intrinsic_matrix.tolist() if self.intrinsic_matrix is not None else None,
            'calibration_error': self.calibration_error,
        }
        
        if hasattr(self, 'intrinsic_params'):
            data['intrinsic_params'] = self.intrinsic_params
        if hasattr(self, 'angle_intrinsic'):
            data['angle_intrinsic'] = self.angle_intrinsic
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"标定参数已保存到: {filename}")
    
    def load_calibration(self, filename):
        """
        从文件加载标定参数
        
        Args:
            filename: 标定文件路径
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.lidar_to_camera = np.array(data['lidar_to_camera']) if data['lidar_to_camera'] else None
        self.camera_to_lidar = np.array(data['camera_to_lidar']) if data['camera_to_lidar'] else None
        self.intrinsic_matrix = np.array(data['camera_intrinsic']) if data['camera_intrinsic'] else None
        self.calibration_error = data.get('calibration_error')
        self.intrinsic_params = data.get('intrinsic_params')
        self.angle_intrinsic = data.get('angle_intrinsic')
        
        print(f"标定参数已从 {filename} 加载")
    
    def get_extrinsic_matrix(self):
        """获取外参矩阵"""
        return self.lidar_to_camera
    
    def get_calibration_error(self):
        """获取标定误差（像素）"""
        return self.calibration_error


class MultiLiDARCalibration:
    """多激光雷达标定类"""
    
    def __init__(self):
        self.lidars = {}  # 存储多个激光雷达的外参
        self.relative_transforms = {}  # 存储激光雷达之间的相对变换
    
    def calibrate_relative(self, lidar1_name, lidar1_points, 
                          lidar2_name, lidar2_points,
                          max_correspondence_distance=0.1):
        """
        标定两个激光雷达之间的相对外参
        
        Args:
            lidar1_name: 激光雷达1名称
            lidar1_points: 激光雷达1点云
            lidar2_name: 激光雷达2名称
            lidar2_points: 激光雷达2点云
            max_correspondence_distance: 最大对应点距离
            
        Returns:
            相对变换矩阵 (lidar2 -> lidar1)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 使用ICP算法
        source = lidar2_points.copy()
        target = lidar1_points.copy()
        
        # 初始变换矩阵
        transform = np.eye(4)
        
        for iteration in range(50):
            # 找最近邻
            nbrs = NearestNeighbors(n_neighbors=1).fit(target)
            distances, indices = nbrs.kneighbors(source)
            
            # 过滤距离过大的对应点
            valid = distances.flatten() < max_correspondence_distance
            
            if np.sum(valid) < 10:
                break
            
            source_valid = source[valid]
            target_valid = target[indices[valid].flatten()]
            
            # 计算变换
            source_centroid = np.mean(source_valid, axis=0)
            target_centroid = np.mean(target_valid, axis=0)
            
            source_centered = source_valid - source_centroid
            target_centered = target_valid - target_centroid
            
            H = np.dot(source_centered.T, target_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            t = target_centroid - np.dot(R, source_centroid)
            
            # 更新变换
            current_transform = np.eye(4)
            current_transform[:3, :3] = R
            current_transform[:3, 3] = t
            
            transform = np.dot(current_transform, transform)
            source = np.dot(source, R.T) + t
        
        # 存储相对变换
        self.relative_transforms[f"{lidar1_name}_{lidar2_name}"] = transform
        
        return transform
    
    def get_relative_transform(self, lidar1_name, lidar2_name):
        """获取两个激光雷达之间的相对变换"""
        key = f"{lidar1_name}_{lidar2_name}"
        return self.relative_transforms.get(key)


def create_synthetic_lidar_points(image_shape, depth_map, camera_intrinsic):
    """
    从深度图生成模拟激光雷达点云
    
    Args:
        image_shape: 图像尺寸 (H, W)
        depth_map: 深度图
        camera_intrinsic: 相机内参
        
    Returns:
        点云 (N, 3)
    """
    h, w = image_shape[:2]
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]
    
    points = []
    
    for v in range(h):
        for u in range(w):
            depth = depth_map[v, u]
            if depth > 0 and depth < 1000:
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                points.append([x, y, z])
    
    return np.array(points)
