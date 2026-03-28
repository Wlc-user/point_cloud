import cv2
import numpy as np
from scipy.spatial import distance as dist

class PrecisionMeasurement:
    def __init__(self):
        self.pixel_to_unit = 1.0
        self.unit = 'pixel'
        self.calibration_points = []
    
    def set_calibration(self, pixel_to_unit, unit='mm'):
        """
        设置校准参数
        
        Args:
            pixel_to_unit: 像素到实际单位的转换因子
            unit: 单位名称
        """
        self.pixel_to_unit = pixel_to_unit
        self.unit = unit
    
    def calibrate_with_reference(self, image, reference_length, **kwargs):
        """
        使用参考物体校准
        
        Args:
            image: 包含参考物体的图像
            reference_length: 参考物体的实际长度
            **kwargs: 参数
            
        Returns:
            是否成功
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        thresh_type = kwargs.get('thresh_type', 'otsu')
        
        from .image_segmentation import ImageSegmenter
        segmenter = ImageSegmenter()
        binary = segmenter.threshold_segmentation(gray, method=thresh_type, **kwargs)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False
        
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        pixel_length = max(width, height)
        
        self.pixel_to_unit = reference_length / pixel_length
        self.calibration_points = [rect]
        
        return True
    
    def measure_distance(self, point1, point2):
        """
        测量两点之间的距离
        
        Args:
            point1: 点1 (x1, y1)
            point2: 点2 (x2, y2)
            
        Returns:
            距离字典
        """
        pixel_distance = dist.euclidean(point1, point2)
        real_distance = pixel_distance * self.pixel_to_unit
        
        return {
            'pixel_distance': pixel_distance,
            'real_distance': real_distance,
            'unit': self.unit
        }
    
    def measure_angle(self, point1, point2, point3):
        """
        测量三点之间的角度 (point2是顶点)
        
        Args:
            point1: 点1
            point2: 顶点
            point3: 点3
            
        Returns:
            角度字典
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        return {
            'angle_radians': angle_rad,
            'angle_degrees': angle_deg
        }
    
    def measure_contour_dimensions(self, contour):
        """
        测量轮廓的尺寸
        
        Args:
            contour: 轮廓
            
        Returns:
            尺寸字典
        """
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        center = rect[0]
        size = rect[1]
        angle = rect[2]
        
        (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
        
        ellipse = None
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
        
        return {
            'bounding_box': {
                'x': x,
                'y': y,
                'width': w * self.pixel_to_unit,
                'height': h * self.pixel_to_unit,
                'width_pixel': w,
                'height_pixel': h
            },
            'min_area_rect': {
                'center': center,
                'size': (size[0] * self.pixel_to_unit, size[1] * self.pixel_to_unit),
                'size_pixel': size,
                'angle': angle
            },
            'min_enclosing_circle': {
                'center': (x_circle, y_circle),
                'radius': radius * self.pixel_to_unit,
                'radius_pixel': radius
            },
            'ellipse': {
                'center': ellipse[0] if ellipse else None,
                'axes': (ellipse[1][0] * self.pixel_to_unit, ellipse[1][1] * self.pixel_to_unit) if ellipse else None,
                'angle': ellipse[2] if ellipse else None
            },
            'area': area * (self.pixel_to_unit ** 2),
            'area_pixel': area,
            'perimeter': perimeter * self.pixel_to_unit,
            'perimeter_pixel': perimeter,
            'unit': self.unit
        }
    
    def measure_circularity(self, contour):
        """
        测量圆度
        
        Args:
            contour: 轮廓
            
        Returns:
            圆度值 (0-1, 1为完美圆形)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity
    
    def measure_holes(self, contour, hierarchy):
        """
        测量孔洞
        
        Args:
            contour: 轮廓
            hierarchy: 层级信息
            
        Returns:
            孔洞信息列表
        """
        holes = []
        
        for i in range(len(contour)):
            if hierarchy[0][i][3] != -1:
                hole_contour = contour[i]
                hole_info = self.measure_contour_dimensions(hole_contour)
                holes.append(hole_info)
        
        return holes
    
    def subpixel_edge_detection(self, image, **kwargs):
        """
        亚像素边缘检测
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            亚像素边缘点
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        threshold1 = kwargs.get('threshold1', 50)
        threshold2 = kwargs.get('threshold2', 150)
        
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        subpixel_contours = []
        for contour in contours:
            if len(contour) >= 6:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                subpixel_contours.append(approx)
        
        return {
            'edges': edges,
            'contours': contours,
            'subpixel_contours': subpixel_contours
        }
    
    def measure_thread_parameters(self, image, pitch_estimate, **kwargs):
        """
        测量螺纹参数
        
        Args:
            image: 螺纹图像
            pitch_estimate: 预估螺距
            **kwargs: 参数
            
        Returns:
            螺纹参数字典
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return None
        
        angles = []
        lengths = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            lengths.append(length)
            angles.append(angle)
        
        avg_angle = np.mean(angles)
        thread_angle = abs(avg_angle)
        
        pitch_profile = cv2.reduce(gray, 0, cv2.REDUCE_AVG).flatten()
        
        peaks = []
        for i in range(1, len(pitch_profile)-1):
            if (pitch_profile[i] > pitch_profile[i-1] and 
                pitch_profile[i] > pitch_profile[i+1]):
                peaks.append(i)
        
        if len(peaks) >= 2:
            peak_distances = np.diff(peaks)
            avg_pitch_pixel = np.mean(peak_distances)
            avg_pitch = avg_pitch_pixel * self.pixel_to_unit
        else:
            avg_pitch = None
        
        return {
            'thread_angle': thread_angle,
            'pitch': avg_pitch,
            'major_diameter': None,
            'minor_diameter': None,
            'pitch_diameter': None
        }
    
    def measure_hole_position(self, image, **kwargs):
        """
        测量孔位
        
        Args:
            image: 图像
            **kwargs: 参数
            
        Returns:
            孔位信息列表
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        min_radius = kwargs.get('min_radius', 10)
        max_radius = kwargs.get('max_radius', 100)
        param1 = kwargs.get('param1', 50)
        param2 = kwargs.get('param2', 30)
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist=20,
                                  param1=param1, param2=param2,
                                  minRadius=min_radius, maxRadius=max_radius)
        
        holes = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
            
            for (x, y, r) in circles:
                holes.append({
                    'center': (x, y),
                    'center_real': (x * self.pixel_to_unit, y * self.pixel_to_unit),
                    'radius': r,
                    'radius_real': r * self.pixel_to_unit,
                    'diameter': 2 * r,
                    'diameter_real': 2 * r * self.pixel_to_unit,
                    'unit': self.unit
                })
        
        return holes
    
    def measure_gear_parameters(self, image, num_teeth_estimate, **kwargs):
        """
        测量齿轮参数
        
        Args:
            image: 齿轮图像
            num_teeth_estimate: 预估齿数
            **kwargs: 参数
            
        Returns:
            齿轮参数字典
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        outer_radius = radius
        
        polar_contour = []
        for point in contour:
            px, py = point[0]
            angle = np.arctan2(py - cy, px - cx)
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            polar_contour.append((angle, dist))
        
        polar_contour = sorted(polar_contour, key=lambda x: x[0])
        distances = np.array([d for _, d in polar_contour])
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(distances, distance=len(distances)//(num_teeth_estimate*2))
        num_teeth = len(peaks)
        
        inner_radius = np.min(distances)
        pitch_diameter = (outer_radius + inner_radius) / 2
        module = pitch_diameter * 2 / num_teeth if num_teeth > 0 else 0
        
        return {
            'center': (cx, cy),
            'center_real': (cx * self.pixel_to_unit, cy * self.pixel_to_unit),
            'outer_radius': outer_radius * self.pixel_to_unit,
            'inner_radius': inner_radius * self.pixel_to_unit,
            'pitch_diameter': pitch_diameter * self.pixel_to_unit,
            'num_teeth': num_teeth,
            'module': module * self.pixel_to_unit,
            'unit': self.unit
        }
    
    def draw_measurement(self, image, measurements, **kwargs):
        """
        绘制测量结果
        
        Args:
            image: 输入图像
            measurements: 测量结果
            **kwargs: 参数
            
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        color = kwargs.get('color', (0, 255, 0))
        thickness = kwargs.get('thickness', 2)
        text_color = kwargs.get('text_color', (255, 0, 0))
        
        if 'bounding_box' in measurements:
            bb = measurements['bounding_box']
            cv2.rectangle(result, (bb['x'], bb['y']),
                        (bb['x'] + int(bb['width_pixel']), bb['y'] + int(bb['height_pixel'])),
                        color, thickness)
        
        if 'min_enclosing_circle' in measurements:
            circle = measurements['min_enclosing_circle']
            cv2.circle(result, (int(circle['center'][0]), int(circle['center'][1])),
                      int(circle['radius_pixel']), color, thickness)
        
        if 'holes' in measurements:
            for hole in measurements['holes']:
                if 'center' in hole and 'radius' in hole:
                    cv2.circle(result, (int(hole['center'][0]), int(hole['center'][1])),
                              int(hole['radius']), color, thickness)
        
        return result
