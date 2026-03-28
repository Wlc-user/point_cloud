import cv2
import numpy as np
from collections import deque

class OpticalFlowAnalyzer:
    def __init__(self, max_history=30):
        self.prev_frame = None
        self.max_history = max_history
        self.flow_history = deque(maxlen=max_history)
        self.magnitude_history = deque(maxlen=max_history)
        self.angle_history = deque(maxlen=max_history)
    
    def calc_optical_flow_farneback(self, current_frame, **kwargs):
        """
        使用Farneback算法计算稠密光流
        
        Args:
            current_frame: 当前帧图像
            **kwargs: 光流计算参数
            
        Returns:
            光流场 (dx, dy)
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return None
        
        # 转换为灰度图
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Farneback算法参数
        pyr_scale = kwargs.get('pyr_scale', 0.5)
        levels = kwargs.get('levels', 3)
        winsize = kwargs.get('winsize', 15)
        iterations = kwargs.get('iterations', 3)
        poly_n = kwargs.get('poly_n', 5)
        poly_sigma = kwargs.get('poly_sigma', 1.2)
        flags = kwargs.get('flags', 0)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, current_gray, None,
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        )
        
        # 更新历史帧
        self.prev_frame = current_gray.copy()
        
        return flow
    
    def calc_optical_flow_lucas_kanade(self, current_frame, feature_params=None, lk_params=None):
        """
        使用Lucas-Kanade算法计算稀疏光流
        
        Args:
            current_frame: 当前帧图像
            feature_params: 特征点检测参数
            lk_params: LK算法参数
            
        Returns:
            特征点位置、新位置、状态
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return None, None, None
        
        # 默认参数
        if feature_params is None:
            feature_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
        
        if lk_params is None:
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点
        p0 = cv2.goodFeaturesToTrack(self.prev_frame, mask=None, **feature_params)
        
        if p0 is None:
            return None, None, None
        
        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, current_gray, p0, None, **lk_params
        )
        
        # 更新历史帧
        self.prev_frame = current_gray.copy()
        
        return p0, p1, st
    
    def analyze_flow_statistics(self, flow, **kwargs):
        """
        统计光流场信息
        
        Args:
            flow: 光流场
            **kwargs: 分析参数
            
        Returns:
            统计信息字典
        """
        if flow is None:
            return None
        
        # 计算光流向量的幅度和角度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 基本统计
        stats = {
            'mean_magnitude': np.mean(mag),
            'std_magnitude': np.std(mag),
            'max_magnitude': np.max(mag),
            'min_magnitude': np.min(mag),
            'mean_angle': np.mean(ang),
            'std_angle': np.std(ang),
            'mean_flow_x': np.mean(flow[..., 0]),
            'mean_flow_y': np.mean(flow[..., 1]),
            'std_flow_x': np.std(flow[..., 0]),
            'std_flow_y': np.std(flow[..., 1])
        }
        
        # 保存历史
        self.flow_history.append(flow)
        self.magnitude_history.append(mag)
        self.angle_history.append(ang)
        
        # 历史统计
        if len(self.magnitude_history) > 1:
            all_mags = np.array(list(self.magnitude_history))
            stats['hist_mean_magnitude'] = np.mean(all_mags)
            stats['hist_std_magnitude'] = np.std(all_mags)
            stats['magnitude_trend'] = self._calculate_trend(list(self.magnitude_history))
        
        return stats
    
    def detect_motion_regions(self, flow, **kwargs):
        """
        检测运动区域
        
        Args:
            flow: 光流场
            **kwargs: 检测参数
            
        Returns:
            运动区域列表
        """
        if flow is None:
            return []
        
        # 计算光流幅度
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 阈值
        threshold = kwargs.get('threshold', 2.0)
        min_area = kwargs.get('min_area', 100)
        
        # 二值化
        _, binary = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
        binary = np.uint8(binary)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小区域
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算区域平均光流
                roi_flow = flow[y:y+h, x:x+w]
                roi_mag = mag[y:y+h, x:x+w]
                
                regions.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'mean_magnitude': np.mean(roi_mag),
                    'mean_flow_x': np.mean(roi_flow[..., 0]),
                    'mean_flow_y': np.mean(roi_flow[..., 1]),
                    'motion_direction': self._get_direction(np.mean(roi_flow[..., 0]), 
                                                          np.mean(roi_flow[..., 1]))
                })
        
        return regions
    
    def detect_anomaly(self, flow, **kwargs):
        """
        基于统计判断检测异常运动
        
        Args:
            flow: 光流场
            **kwargs: 检测参数
            
        Returns:
            异常检测结果
        """
        if flow is None or len(self.magnitude_history) < 5:
            return {'is_anomaly': False, 'confidence': 0.0}
        
        # 当前统计
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        current_mean = np.mean(mag)
        current_std = np.std(mag)
        
        # 历史统计
        hist_means = [np.mean(m) for m in self.magnitude_history]
        hist_mean = np.mean(hist_means)
        hist_std = np.std(hist_means)
        
        # 异常判断阈值
        threshold_sigma = kwargs.get('threshold_sigma', 3.0)
        
        # Z-score判断
        if hist_std > 0:
            z_score = abs(current_mean - hist_mean) / hist_std
            is_anomaly = z_score > threshold_sigma
            confidence = min(z_score / threshold_sigma, 1.0)
        else:
            is_anomaly = False
            confidence = 0.0
        
        # 额外的异常检测规则
        anomaly_rules = []
        
        # 规则1: 幅度突变
        if len(self.magnitude_history) >= 2:
            prev_mean = np.mean(list(self.magnitude_history)[-2])
            if prev_mean > 0 and abs(current_mean - prev_mean) / prev_mean > 0.5:
                anomaly_rules.append('magnitude_spike')
        
        # 规则2: 方向突变
        if len(self.angle_history) >= 2:
            current_ang = np.mean(list(self.angle_history)[-1])
            prev_ang = np.mean(list(self.angle_history)[-2])
            if abs(current_ang - prev_ang) > np.pi / 4:  # 45度
                anomaly_rules.append('direction_change')
        
        # 规则3: 光流分布异常
        if current_std > hist_std * 2:
            anomaly_rules.append('distribution_anomaly')
        
        return {
            'is_anomaly': is_anomaly or len(anomaly_rules) > 0,
            'confidence': confidence,
            'z_score': z_score if hist_std > 0 else 0,
            'anomaly_rules': anomaly_rules,
            'current_mean': current_mean,
            'hist_mean': hist_mean,
            'current_std': current_std,
            'hist_std': hist_std
        }
    
    def classify_motion_pattern(self, flow, **kwargs):
        """
        基于统计特征分类运动模式
        
        Args:
            flow: 光流场
            **kwargs: 分类参数
            
        Returns:
            运动模式分类结果
        """
        if flow is None:
            return {'pattern': 'unknown', 'confidence': 0.0}
        
        # 计算光流特征
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        mean_mag = np.mean(mag)
        std_mag = np.std(mag)
        mean_ang = np.mean(ang)
        std_ang = np.std(ang)
        
        # 运动模式判断
        patterns = []
        
        # 模式1: 静止
        if mean_mag < 0.5:
            patterns.append(('static', 1.0))
        
        # 模式2: 匀速运动
        elif std_mag < mean_mag * 0.2 and std_ang < 0.3:
            patterns.append(('uniform_motion', 0.9))
        
        # 模式3: 加速/减速
        elif std_mag > mean_mag * 0.5:
            patterns.append(('acceleration', 0.8))
        
        # 模式4: 旋转
        elif std_ang > 1.0:
            patterns.append(('rotation', 0.7))
        
        # 模式5: 振动
        elif std_mag > mean_mag * 0.3 and mean_mag < 5.0:
            patterns.append(('vibration', 0.6))
        
        # 模式6: 随机运动
        else:
            patterns.append(('random', 0.5))
        
        # 选择最可能的模式
        best_pattern = max(patterns, key=lambda x: x[1])
        
        return {
            'pattern': best_pattern[0],
            'confidence': best_pattern[1],
            'all_patterns': patterns,
            'features': {
                'mean_magnitude': mean_mag,
                'std_magnitude': std_mag,
                'mean_angle': mean_ang,
                'std_angle': std_ang
            }
        }
    
    def estimate_velocity(self, flow, pixel_scale=1.0, fps=30.0):
        """
        估计运动速度
        
        Args:
            flow: 光流场
            pixel_scale: 像素到实际单位的转换比例
            fps: 帧率
            
        Returns:
            速度估计结果
        """
        if flow is None:
            return {'velocity_x': 0, 'velocity_y': 0, 'velocity_magnitude': 0, 'velocity_angle': 0}
        
        # 计算平均光流
        mean_flow_x = np.mean(flow[..., 0])
        mean_flow_y = np.mean(flow[..., 1])
        
        # 转换为实际速度
        velocity_x = mean_flow_x * pixel_scale * fps
        velocity_y = mean_flow_y * pixel_scale * fps
        
        # 计算速度大小和方向
        velocity_mag = np.sqrt(velocity_x**2 + velocity_y**2)
        velocity_ang = np.arctan2(velocity_y, velocity_x)
        
        return {
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_magnitude': velocity_mag,
            'velocity_angle': velocity_ang,
            'direction': self._get_direction(velocity_x, velocity_y)
        }
    
    def visualize_flow(self, frame, flow, **kwargs):
        """
        可视化光流场
        
        Args:
            frame: 原始帧
            flow: 光流场
            **kwargs: 可视化参数
            
        Returns:
            可视化图像
        """
        if flow is None:
            return frame
        
        # 计算光流HSV表示
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 创建HSV图像
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255  # 饱和度
        hsv[..., 0] = ang * 180 / np.pi / 2  # 色调表示方向
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 亮度表示幅度
        
        # 转换为BGR
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 叠加模式
        overlay = kwargs.get('overlay', False)
        if overlay:
            alpha = kwargs.get('alpha', 0.5)
            result = cv2.addWeighted(frame, 1-alpha, flow_vis, alpha, 0)
        else:
            result = flow_vis
        
        return result
    
    def draw_flow_arrows(self, frame, flow, step=16, **kwargs):
        """
        绘制光流向量箭头
        
        Args:
            frame: 原始帧
            flow: 光流场
            step: 采样步长
            **kwargs: 绘制参数
            
        Returns:
            绘制结果
        """
        if flow is None:
            return frame
        
        result = frame.copy()
        h, w = flow.shape[:2]
        
        # 采样点
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
        
        # 获取光流向量
        fx, fy = flow[y, x].T
        
        # 过滤小幅度
        min_mag = kwargs.get('min_magnitude', 1.0)
        mag = np.sqrt(fx**2 + fy**2)
        mask = mag > min_mag
        
        x, y, fx, fy = x[mask], y[mask], fx[mask], fy[mask]
        
        # 绘制箭头
        color = kwargs.get('color', (0, 255, 0))
        thickness = kwargs.get('thickness', 1)
        
        for i in range(len(x)):
            cv2.arrowedLine(result, (x[i], y[i]), 
                          (int(x[i] + fx[i]), int(y[i] + fy[i])),
                          color, thickness)
        
        return result
    
    def reset(self):
        """
        重置分析器状态
        """
        self.prev_frame = None
        self.flow_history.clear()
        self.magnitude_history.clear()
        self.angle_history.clear()
    
    def _calculate_trend(self, values):
        """
        计算趋势
        """
        if len(values) < 2:
            return 0.0
        
        # 线性回归
        x = np.arange(len(values))
        y = np.array([np.mean(v) for v in values])
        
        # 计算斜率
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def _get_direction(self, dx, dy):
        """
        获取方向描述
        """
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if -22.5 <= angle < 22.5:
            return 'right'
        elif 22.5 <= angle < 67.5:
            return 'down_right'
        elif 67.5 <= angle < 112.5:
            return 'down'
        elif 112.5 <= angle < 157.5:
            return 'down_left'
        elif -157.5 <= angle < -112.5:
            return 'up_left'
        elif -112.5 <= angle < -67.5:
            return 'up'
        elif -67.5 <= angle < -22.5:
            return 'up_right'
        else:
            return 'left'