import cv2
import numpy as np

class ImageAnalyzer:
    def __init__(self):
        pass
    
    def detect_objects(self, image, **kwargs):
        """
        目标检测
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            检测到的目标列表，每个目标包含位置和尺寸
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 应用阈值处理
        thresh = kwargs.get('thresh', 127)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        min_area = kwargs.get('min_area', 100)
        objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area
                })
        
        return objects
    
    def measure_dimensions(self, image, objects, **kwargs):
        """
        测量目标尺寸
        
        Args:
            image: 输入图像
            objects: 目标列表
            **kwargs: 测量参数
            
        Returns:
            带尺寸信息的目标列表
        """
        calibrated_objects = []
        
        # 像素到实际单位的转换因子
        pixel_to_unit = kwargs.get('pixel_to_unit', 1.0)
        unit = kwargs.get('unit', 'px')
        
        for obj in objects:
            # 计算实际尺寸
            real_width = obj['width'] * pixel_to_unit
            real_height = obj['height'] * pixel_to_unit
            real_area = obj['area'] * (pixel_to_unit ** 2)
            
            calibrated_objects.append({
                **obj,
                'real_width': real_width,
                'real_height': real_height,
                'real_area': real_area,
                'unit': unit
            })
        
        return calibrated_objects
    
    def detect_defects(self, image, **kwargs):
        """
        缺陷检测
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            缺陷列表
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 应用高斯滤波
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        min_area = kwargs.get('min_area', 50)
        defects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'type': 'defect'
                })
        
        return defects
    
    def calculate_color_histogram(self, image, **kwargs):
        """
        计算颜色直方图
        
        Args:
            image: 输入图像
            **kwargs: 计算参数
            
        Returns:
            颜色直方图
        """
        channels = kwargs.get('channels', [0, 1, 2])
        hist_size = kwargs.get('hist_size', [256, 256, 256])
        ranges = kwargs.get('ranges', [0, 256, 0, 256, 0, 256])
        
        hist = cv2.calcHist([image], channels, None, hist_size, ranges)
        return hist
    
    def compare_images(self, image1, image2, **kwargs):
        """
        比较两个图像
        
        Args:
            image1: 第一个图像
            image2: 第二个图像
            **kwargs: 比较参数
            
        Returns:
            相似度分数
        """
        # 调整图像大小
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 计算差异
        diff = cv2.absdiff(image1, image2)
        
        # 计算相似度
        similarity = 1.0 - (np.sum(diff) / (image1.size * 255))
        
        return similarity
    
    def ocr_detection(self, image, **kwargs):
        """
        OCR文字检测
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            检测到的文字列表
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 应用阈值处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤文字区域
        min_width = kwargs.get('min_width', 10)
        min_height = kwargs.get('min_height', 10)
        max_width = kwargs.get('max_width', 200)
        max_height = kwargs.get('max_height', 50)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w >= min_width and h >= min_height and 
                w <= max_width and h <= max_height):
                text_regions.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })
        
        return text_regions
    
    def shape_recognition(self, image, **kwargs):
        """
        形状识别
        
        Args:
            image: 输入图像
            **kwargs: 识别参数
            
        Returns:
            识别到的形状列表
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 应用阈值处理
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            # 近似轮廓
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 识别形状
            shape = 'unknown'
            if len(approx) == 3:
                shape = 'triangle'
            elif len(approx) == 4:
                # 检查是否为矩形
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = 'square'
                else:
                    shape = 'rectangle'
            elif len(approx) > 4:
                # 检查是否为圆形
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.7:
                        shape = 'circle'
            
            x, y, w, h = cv2.boundingRect(contour)
            shapes.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'shape': shape
            })
        
        return shapes