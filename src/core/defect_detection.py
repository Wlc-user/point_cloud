import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, feature
from skimage.filters import threshold_otsu, threshold_local

class DefectDetector:
    def __init__(self):
        pass
    
    def detect_scratches(self, image, **kwargs):
        """
        检测表面划痕缺陷
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            划痕检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用Canny边缘检测
        low_threshold = kwargs.get('low_threshold', 50)
        high_threshold = kwargs.get('high_threshold', 150)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # 形态学操作连接断续边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤划痕特征
        min_length = kwargs.get('min_length', 50)
        max_width = kwargs.get('max_width', 10)
        min_aspect_ratio = kwargs.get('min_aspect_ratio', 5.0)
        
        scratches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)
            aspect_ratio = length / width if width > 0 else 0
            
            if length >= min_length and width <= max_width and aspect_ratio >= min_aspect_ratio:
                scratches.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'length': length,
                    'width': width,
                    'aspect_ratio': aspect_ratio,
                    'type': 'scratch'
                })
        
        return scratches
    
    def detect_dents(self, image, **kwargs):
        """
        检测凹陷缺陷
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            凹陷检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用局部阈值检测暗区域
        block_size = kwargs.get('block_size', 15)
        thresh = threshold_local(gray, block_size, offset=10)
        binary = gray < thresh
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤凹陷特征
        min_area = kwargs.get('min_area', 100)
        max_area = kwargs.get('max_area', 5000)
        circularity_threshold = kwargs.get('circularity_threshold', 0.5)
        
        dents = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if circularity >= circularity_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    dents.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'circularity': circularity,
                        'type': 'dent'
                    })
        
        return dents
    
    def detect_bumps(self, image, **kwargs):
        """
        检测凸起缺陷
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            凸起检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用高斯差分检测凸起
        sigma1 = kwargs.get('sigma1', 2)
        sigma2 = kwargs.get('sigma2', 5)
        
        blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
        dog = blur1 - blur2
        
        # 阈值处理
        threshold = kwargs.get('threshold', 30)
        _, binary = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤凸起特征
        min_area = kwargs.get('min_area', 50)
        max_area = kwargs.get('max_area', 2000)
        
        bumps = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                bumps.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'type': 'bump'
                })
        
        return bumps
    
    def detect_stains(self, image, **kwargs):
        """
        检测污渍缺陷
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            污渍检测结果
        """
        if len(image.shape) == 3:
            # 转换到HSV空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        
        # 颜色范围参数
        lower_h = kwargs.get('lower_h', 0)
        lower_s = kwargs.get('lower_s', 0)
        lower_v = kwargs.get('lower_v', 0)
        upper_h = kwargs.get('upper_h', 180)
        upper_s = kwargs.get('upper_s', 255)
        upper_v = kwargs.get('upper_v', 100)
        
        # 创建颜色掩码
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤污渍特征
        min_area = kwargs.get('min_area', 100)
        
        stains = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算平均颜色
                roi = hsv[y:y+h, x:x+w]
                avg_color = np.mean(roi, axis=(0, 1))
                
                stains.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'avg_color': avg_color,
                    'type': 'stain'
                })
        
        return stains
    
    def detect_color_defects(self, image, reference_image=None, **kwargs):
        """
        检测颜色缺陷（色差、褪色等）
        
        Args:
            image: 输入图像
            reference_image: 参考图像
            **kwargs: 检测参数
            
        Returns:
            颜色缺陷检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if reference_image is not None:
            if len(reference_image.shape) == 3:
                ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = reference_image
            
            # 计算差异
            diff = cv2.absdiff(gray, ref_gray)
        else:
            # 使用局部标准差检测颜色变化
            kernel_size = kwargs.get('kernel_size', 15)
            mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            diff = cv2.absdiff(gray, mean)
        
        # 阈值处理
        threshold = kwargs.get('threshold', 30)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤缺陷区域
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
                    'type': 'color_defect'
                })
        
        return defects
    
    def detect_texture_defects(self, image, **kwargs):
        """
        检测纹理缺陷
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            纹理缺陷检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用LBP（局部二值模式）检测纹理变化
        radius = kwargs.get('radius', 3)
        n_points = kwargs.get('n_points', 8 * radius)
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # 计算LBP直方图
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        
        # 使用Otsu阈值检测异常区域
        threshold = kwargs.get('threshold', 0.5)
        normalized_hist = hist.astype(float) / hist.sum()
        
        # 检测异常纹理区域
        block_size = kwargs.get('block_size', 32)
        h, w = gray.shape
        
        defects = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_lbp = lbp[i:i+block_size, j:j+block_size]
                block_hist, _ = np.histogram(block_lbp.ravel(), 
                                          bins=n_points + 2, 
                                          range=(0, n_points + 2))
                block_hist = block_hist.astype(float) / block_hist.sum()
                
                # 计算直方图差异
                diff = np.sum(np.abs(normalized_hist - block_hist))
                
                if diff > threshold:
                    defects.append({
                        'x': j,
                        'y': i,
                        'width': block_size,
                        'height': block_size,
                        'texture_diff': diff,
                        'type': 'texture_defect'
                    })
        
        return defects
    
    def detect_edge_defects(self, image, **kwargs):
        """
        检测边缘缺陷（毛刺、缺口等）
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            边缘缺陷检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 边缘检测
        low_threshold = kwargs.get('low_threshold', 50)
        high_threshold = kwargs.get('high_threshold', 150)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 检测边缘缺陷
        min_length = kwargs.get('min_length', 10)
        max_length = kwargs.get('max_length', 100)
        
        defects = []
        for contour in contours:
            # 计算轮廓长度
            perimeter = cv2.arcLength(contour, True)
            
            # 计算凸包
            hull = cv2.convexHull(contour)
            hull_perimeter = cv2.arcLength(hull, True)
            
            # 凸性缺陷
            convexity = perimeter / hull_perimeter if hull_perimeter > 0 else 0
            
            # 检测毛刺（凸性缺陷）
            if convexity > 1.1 and min_length <= perimeter <= max_length:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'perimeter': perimeter,
                    'convexity': convexity,
                    'type': 'burr'
                })
            
            # 检测缺口（凹性缺陷）
            elif convexity < 0.9 and min_length <= perimeter <= max_length:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'perimeter': perimeter,
                    'convexity': convexity,
                    'type': 'notch'
                })
        
        return defects
    
    def detect_missing_parts(self, image, template_image, **kwargs):
        """
        检测缺失部件
        
        Args:
            image: 输入图像
            template_image: 模板图像
            **kwargs: 检测参数
            
        Returns:
            缺失部件检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if len(template_image.shape) == 3:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_image
        
        # 模板匹配
        method = kwargs.get('method', cv2.TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(gray, template_gray, method)
        
        # 查找匹配位置
        threshold = kwargs.get('threshold', 0.8)
        locations = np.where(result >= threshold)
        
        missing_parts = []
        if len(locations[0]) == 0:
            # 没有找到匹配，可能缺失
            missing_parts.append({
                'type': 'missing_part',
                'confidence': 0.0,
                'message': 'Template not found'
            })
        else:
            # 找到匹配，检查匹配质量
            max_val = np.max(result)
            if max_val < threshold:
                missing_parts.append({
                    'type': 'missing_part',
                    'confidence': max_val,
                    'message': 'Low confidence match'
                })
        
        return missing_parts
    
    def detect_welding_defects(self, image, **kwargs):
        """
        检测焊接缺陷（气孔、裂纹等）
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            焊接缺陷检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 检测气孔（圆形暗点）
        min_radius = kwargs.get('min_radius', 2)
        max_radius = kwargs.get('max_radius', 10)
        
        # 使用Hough圆检测
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_radius*2,
            param1=50, param2=30,
            minRadius=min_radius, maxRadius=max_radius
        )
        
        defects = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                defects.append({
                    'x': x,
                    'y': y,
                    'radius': r,
                    'type': 'porosity'
                })
        
        # 检测裂纹（线性特征）
        edges = cv2.Canny(enhanced, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                              threshold=50, 
                              minLineLength=20, 
                              maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 20:
                    defects.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'length': length,
                        'type': 'crack'
                    })
        
        return defects
    
    def detect_pcb_defects(self, image, **kwargs):
        """
        检测PCB缺陷（短路、开路等）
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
            
        Returns:
            PCB缺陷检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 检测短路（不应该连接的线路连接）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        min_area = kwargs.get('min_area', 100)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area * 5:  # 异常大面积可能表示短路
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'type': 'short_circuit'
                })
        
        # 检测开路（线路中断）
        eroded = cv2.erode(binary, kernel, iterations=2)
        contours2, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours2:
            area = cv2.contourArea(contour)
            if area < min_area / 2:  # 异常小面积可能表示开路
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'type': 'open_circuit'
                })
        
        return defects
    
    def detect_dimensional_defects(self, image, expected_dimensions, **kwargs):
        """
        检测尺寸缺陷
        
        Args:
            image: 输入图像
            expected_dimensions: 期望尺寸 {'width': w, 'height': h}
            **kwargs: 检测参数
            
        Returns:
            尺寸缺陷检测结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        tolerance = kwargs.get('tolerance', 0.1)  # 10% 容差
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查宽度
            expected_width = expected_dimensions.get('width')
            if expected_width:
                width_error = abs(w - expected_width) / expected_width
                if width_error > tolerance:
                    defects.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'expected_width': expected_width,
                        'actual_width': w,
                        'width_error': width_error,
                        'type': 'width_defect'
                    })
            
            # 检查高度
            expected_height = expected_dimensions.get('height')
            if expected_height:
                height_error = abs(h - expected_height) / expected_height
                if height_error > tolerance:
                    defects.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'expected_height': expected_height,
                        'actual_height': h,
                        'height_error': height_error,
                        'type': 'height_defect'
                    })
        
        return defects
    
    def visualize_defects(self, image, defects, **kwargs):
        """
        可视化缺陷检测结果
        
        Args:
            image: 输入图像
            defects: 缺陷列表
            **kwargs: 可视化参数
            
        Returns:
            可视化图像
        """
        result = image.copy()
        
        # 缺陷颜色映射
        color_map = {
            'scratch': (0, 0, 255),      # 红色
            'dent': (255, 0, 0),        # 蓝色
            'bump': (0, 255, 255),      # 青色
            'stain': (0, 165, 255),     # 橙色
            'color_defect': (255, 0, 255), # 紫色
            'texture_defect': (255, 255, 0), # 黄色
            'burr': (0, 255, 0),        # 绿色
            'notch': (128, 0, 128),      # 紫红色
            'porosity': (255, 100, 0),    # 深橙色
            'crack': (0, 100, 255),      # 橙红色
            'short_circuit': (255, 0, 100), # 粉红色
            'open_circuit': (100, 100, 100), # 灰色
            'width_defect': (255, 255, 255), # 白色
            'height_defect': (200, 200, 200)  # 浅灰色
        }
        
        thickness = kwargs.get('thickness', 2)
        show_label = kwargs.get('show_label', True)
        
        for defect in defects:
            defect_type = defect.get('type', 'unknown')
            color = color_map.get(defect_type, (0, 255, 0))
            
            # 绘制缺陷边界框
            x = defect.get('x', 0)
            y = defect.get('y', 0)
            w = defect.get('width', 0)
            h = defect.get('height', 0)
            
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
            
            # 添加标签
            if show_label:
                label = f"{defect_type}"
                cv2.putText(result, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result