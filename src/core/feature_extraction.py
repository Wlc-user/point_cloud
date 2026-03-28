import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, greycomatrix, greycoprops

class FeatureExtractor:
    def __init__(self):
        pass
    
    def detect_keypoints(self, image, method='orb', **kwargs):
        """
        检测关键点
        
        Args:
            image: 输入图像
            method: 检测方法 (orb, sift, surf, fast, harris)
            **kwargs: 检测参数
            
        Returns:
            关键点列表
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'orb':
            n_features = kwargs.get('n_features', 500)
            scale_factor = kwargs.get('scale_factor', 1.2)
            n_levels = kwargs.get('n_levels', 8)
            orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)
            keypoints = orb.detect(gray, None)
            return keypoints
        
        elif method == 'sift':
            n_features = kwargs.get('n_features', 500)
            n_octave_layers = kwargs.get('n_octave_layers', 3)
            sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers)
            keypoints = sift.detect(gray, None)
            return keypoints
        
        elif method == 'surf':
            hessian_threshold = kwargs.get('hessian_threshold', 400)
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
            keypoints = surf.detect(gray, None)
            return keypoints
        
        elif method == 'fast':
            threshold = kwargs.get('threshold', 20)
            nonmax_suppression = kwargs.get('nonmax_suppression', True)
            fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
            keypoints = fast.detect(gray, None)
            return keypoints
        
        elif method == 'harris':
            block_size = kwargs.get('block_size', 2)
            ksize = kwargs.get('ksize', 3)
            k = kwargs.get('k', 0.04)
            threshold = kwargs.get('threshold', 0.01)
            
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, block_size, ksize, k)
            dst = cv2.dilate(dst, None)
            
            keypoints = []
            y, x = np.where(dst > threshold * dst.max())
            for i, j in zip(x, y):
                keypoints.append(cv2.KeyPoint(float(i), float(j), 1))
            
            return keypoints
        
        else:
            raise ValueError(f"不支持的关键点检测方法: {method}")
    
    def compute_descriptors(self, image, keypoints, method='orb', **kwargs):
        """
        计算描述子
        
        Args:
            image: 输入图像
            keypoints: 关键点列表
            method: 描述子方法 (orb, sift, surf, brief, freak)
            **kwargs: 计算参数
            
        Returns:
            描述子数组
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'orb':
            n_features = kwargs.get('n_features', 500)
            orb = cv2.ORB_create(nfeatures=n_features)
            keypoints, descriptors = orb.compute(gray, keypoints)
            return keypoints, descriptors
        
        elif method == 'sift':
            n_features = kwargs.get('n_features', 500)
            sift = cv2.SIFT_create(nfeatures=n_features)
            keypoints, descriptors = sift.compute(gray, keypoints)
            return keypoints, descriptors
        
        elif method == 'surf':
            hessian_threshold = kwargs.get('hessian_threshold', 400)
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
            keypoints, descriptors = surf.compute(gray, keypoints)
            return keypoints, descriptors
        
        elif method == 'brief':
            bytes = kwargs.get('bytes', 32)
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=bytes)
            keypoints, descriptors = brief.compute(gray, keypoints)
            return keypoints, descriptors
        
        elif method == 'freak':
            freak = cv2.xfeatures2d.FREAK_create()
            keypoints, descriptors = freak.compute(gray, keypoints)
            return keypoints, descriptors
        
        else:
            raise ValueError(f"不支持的描述子方法: {method}")
    
    def detect_and_compute(self, image, method='orb', **kwargs):
        """
        检测关键点并计算描述子
        
        Args:
            image: 输入图像
            method: 方法 (orb, sift, surf)
            **kwargs: 参数
            
        Returns:
            (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'orb':
            n_features = kwargs.get('n_features', 500)
            orb = cv2.ORB_create(nfeatures=n_features)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            return keypoints, descriptors
        
        elif method == 'sift':
            n_features = kwargs.get('n_features', 500)
            sift = cv2.SIFT_create(nfeatures=n_features)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            return keypoints, descriptors
        
        elif method == 'surf':
            hessian_threshold = kwargs.get('hessian_threshold', 400)
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
            keypoints, descriptors = surf.detectAndCompute(gray, None)
            return keypoints, descriptors
        
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def draw_keypoints(self, image, keypoints, **kwargs):
        """
        绘制关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点列表
            **kwargs: 绘制参数
            
        Returns:
            绘制后的图像
        """
        color = kwargs.get('color', (0, 255, 0))
        flags = kwargs.get('flags', cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        result = image.copy()
        return cv2.drawKeypoints(result, keypoints, None, color=color, flags=flags)
    
    def extract_hog_features(self, image, **kwargs):
        """
        提取HOG特征
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            HOG特征
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        orientations = kwargs.get('orientations', 9)
        pixels_per_cell = kwargs.get('pixels_per_cell', (8, 8))
        cells_per_block = kwargs.get('cells_per_block', (2, 2))
        visualize = kwargs.get('visualize', False)
        
        if visualize:
            features, hog_image = hog(gray, orientations=orientations,
                                      pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block,
                                      visualize=True)
            return features, hog_image
        else:
            features = hog(gray, orientations=orientations,
                         pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block,
                         visualize=False)
            return features
    
    def extract_lbp_features(self, image, **kwargs):
        """
        提取LBP特征
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            LBP特征
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        radius = kwargs.get('radius', 3)
        n_points = kwargs.get('n_points', 8 * radius)
        method = kwargs.get('method', 'uniform')
        
        lbp = local_binary_pattern(gray, n_points, radius, method=method)
        
        n_bins = n_points + 2 if method == 'uniform' else 2 ** n_points
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-7)
        
        return hist, lbp
    
    def extract_glcm_features(self, image, **kwargs):
        """
        提取GLCM特征
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            GLCM特征字典
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        distances = kwargs.get('distances', [1])
        angles = kwargs.get('angles', [0, np.pi/4, np.pi/2, 3*np.pi/4])
        levels = kwargs.get('levels', 256)
        
        glcm = greycomatrix(gray, distances=distances, angles=angles, 
                           levels=levels, symmetric=True, normed=True)
        
        features = {}
        features['contrast'] = greycoprops(glcm, 'contrast').mean()
        features['dissimilarity'] = greycoprops(glcm, 'dissimilarity').mean()
        features['homogeneity'] = greycoprops(glcm, 'homogeneity').mean()
        features['energy'] = greycoprops(glcm, 'energy').mean()
        features['correlation'] = greycoprops(glcm, 'correlation').mean()
        features['ASM'] = greycoprops(glcm, 'ASM').mean()
        
        return features
    
    def extract_color_features(self, image, **kwargs):
        """
        提取颜色特征
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            颜色特征字典
        """
        if len(image.shape) != 3:
            raise ValueError("输入必须是彩色图像")
        
        features = {}
        
        channels = cv2.split(image)
        for i, channel in enumerate(['B', 'G', 'R']):
            features[f'{channel}_mean'] = np.mean(channel)
            features[f'{channel}_std'] = np.std(channel)
            features[f'{channel}_skewness'] = np.mean(((channel - features[f'{channel}_mean']) / (features[f'{channel}_std'] + 1e-7)) ** 3)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_channels = cv2.split(hsv)
        for i, channel in enumerate(['H', 'S', 'V']):
            features[f'{channel}_mean'] = np.mean(hsv_channels[i])
            features[f'{channel}_std'] = np.std(hsv_channels[i])
        
        hist_bgr = []
        for channel in channels:
            hist, _ = np.histogram(channel.ravel(), bins=32, range=(0, 256))
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-7)
            hist_bgr.extend(hist)
        
        features['color_histogram'] = np.array(hist_bgr)
        
        return features
    
    def extract_shape_features(self, contour, **kwargs):
        """
        提取形状特征
        
        Args:
            contour: 轮廓
            **kwargs: 参数
            
        Returns:
            形状特征字典
        """
        features = {}
        
        area = cv2.contourArea(contour)
        features['area'] = area
        
        perimeter = cv2.arcLength(contour, True)
        features['perimeter'] = perimeter
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            features['circularity'] = circularity
        
        x, y, w, h = cv2.boundingRect(contour)
        features['aspect_ratio'] = float(w) / h if h > 0 else 0
        features['extent'] = area / (w * h) if w * h > 0 else 0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = area / hull_area if hull_area > 0 else 0
        
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            features['centroid'] = (cx, cy)
            
            hu_moments = cv2.HuMoments(M).flatten()
            for i, hu in enumerate(hu_moments):
                features[f'hu_{i+1}'] = hu
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        features['min_area_rect_angle'] = rect[2]
        features['min_area_rect_size'] = rect[1]
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        features['min_enclosing_circle_radius'] = radius
        
        ellipse = cv2.fitEllipse(contour)
        features['ellipse_center'] = ellipse[0]
        features['ellipse_axes'] = ellipse[1]
        features['ellipse_angle'] = ellipse[2]
        
        return features
    
    def extract_all_features(self, image, contour=None, **kwargs):
        """
        提取所有特征
        
        Args:
            image: 输入图像
            contour: 可选的轮廓
            **kwargs: 参数
            
        Returns:
            特征字典
        """
        features = {}
        
        if kwargs.get('color', True):
            features['color'] = self.extract_color_features(image)
        
        if kwargs.get('texture', True):
            features['lbp'], _ = self.extract_lbp_features(image)
            features['glcm'] = self.extract_glcm_features(image)
        
        if kwargs.get('shape', True) and contour is not None:
            features['shape'] = self.extract_shape_features(contour)
        
        if kwargs.get('hog', True):
            features['hog'] = self.extract_hog_features(image)
        
        return features
