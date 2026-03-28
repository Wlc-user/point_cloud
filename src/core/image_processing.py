import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass
    
    def apply_filter(self, image, filter_type, **kwargs):
        """
        应用各种滤波操作
        
        Args:
            image: 输入图像
            filter_type: 滤波类型 (gaussian, median, bilateral, average)
            **kwargs: 滤波参数
            
        Returns:
            滤波后的图像
        """
        if filter_type == 'gaussian':
            kernel_size = kwargs.get('kernel_size', (5, 5))
            sigma_x = kwargs.get('sigma_x', 0)
            return cv2.GaussianBlur(image, kernel_size, sigma_x)
        elif filter_type == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            return cv2.medianBlur(image, kernel_size)
        elif filter_type == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        elif filter_type == 'average':
            kernel_size = kwargs.get('kernel_size', (5, 5))
            return cv2.blur(image, kernel_size)
        else:
            raise ValueError(f"不支持的滤波类型: {filter_type}")
    
    def detect_edges(self, image, method='canny', **kwargs):
        """
        边缘检测
        
        Args:
            image: 输入图像
            method: 边缘检测方法 (canny, sobel, laplacian)
            **kwargs: 检测参数
            
        Returns:
            边缘图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'canny':
            threshold1 = kwargs.get('threshold1', 100)
            threshold2 = kwargs.get('threshold2', 200)
            return cv2.Canny(gray, threshold1, threshold2)
        elif method == 'sobel':
            dx = kwargs.get('dx', 1)
            dy = kwargs.get('dy', 1)
            ksize = kwargs.get('ksize', 3)
            return cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        elif method == 'laplacian':
            ksize = kwargs.get('ksize', 3)
            return cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        else:
            raise ValueError(f"不支持的边缘检测方法: {method}")
    
    def morphological_operations(self, image, operation, **kwargs):
        """
        形态学操作
        
        Args:
            image: 输入图像
            operation: 操作类型 (erosion, dilation, opening, closing)
            **kwargs: 操作参数
            
        Returns:
            操作后的图像
        """
        kernel = kwargs.get('kernel', np.ones((3, 3), np.uint8))
        iterations = kwargs.get('iterations', 1)
        
        if operation == 'erosion':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilation':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError(f"不支持的形态学操作: {operation}")
    
    def threshold(self, image, method='binary', **kwargs):
        """
        阈值处理
        
        Args:
            image: 输入图像
            method: 阈值方法 (binary, adaptive, otsu)
            **kwargs: 阈值参数
            
        Returns:
            二值化图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'binary':
            thresh = kwargs.get('thresh', 127)
            maxval = kwargs.get('maxval', 255)
            return cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)[1]
        elif method == 'adaptive':
            maxval = kwargs.get('maxval', 255)
            adaptive_method = kwargs.get('adaptive_method', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            threshold_type = kwargs.get('threshold_type', cv2.THRESH_BINARY)
            block_size = kwargs.get('block_size', 11)
            C = kwargs.get('C', 2)
            return cv2.adaptiveThreshold(gray, maxval, adaptive_method, threshold_type, block_size, C)
        elif method == 'otsu':
            maxval = kwargs.get('maxval', 255)
            return cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            raise ValueError(f"不支持的阈值方法: {method}")
    
    def find_contours(self, image, **kwargs):
        """
        查找轮廓
        
        Args:
            image: 输入图像
            **kwargs: 查找参数
            
        Returns:
            轮廓列表
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mode = kwargs.get('mode', cv2.RETR_EXTERNAL)
        method = kwargs.get('method', cv2.CHAIN_APPROX_SIMPLE)
        
        return cv2.findContours(gray, mode, method)[0]
    
    def draw_contours(self, image, contours, **kwargs):
        """
        绘制轮廓
        
        Args:
            image: 输入图像
            contours: 轮廓列表
            **kwargs: 绘制参数
            
        Returns:
            绘制后的图像
        """
        color = kwargs.get('color', (0, 255, 0))
        thickness = kwargs.get('thickness', 2)
        
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    def resize(self, image, **kwargs):
        """
        调整图像大小
        
        Args:
            image: 输入图像
            **kwargs: 调整参数
            
        Returns:
            调整后的图像
        """
        width = kwargs.get('width', None)
        height = kwargs.get('height', None)
        scale = kwargs.get('scale', None)
        
        if scale is not None:
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
        elif width is not None and height is not None:
            new_width = width
            new_height = height
        else:
            raise ValueError("必须指定width和height或scale参数")
        
        return cv2.resize(image, (new_width, new_height))
    
    def crop(self, image, x, y, width, height):
        """
        裁剪图像
        
        Args:
            image: 输入图像
            x: 起始x坐标
            y: 起始y坐标
            width: 裁剪宽度
            height: 裁剪高度
            
        Returns:
            裁剪后的图像
        """
        return image[y:y+height, x:x+width]
    
    def rotate(self, image, angle, **kwargs):
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度
            **kwargs: 旋转参数
            
        Returns:
            旋转后的图像
        """
        center = kwargs.get('center', (image.shape[1]//2, image.shape[0]//2))
        scale = kwargs.get('scale', 1.0)
        
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))