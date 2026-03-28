import cv2
import numpy as np
from scipy import ndimage

class ImageSegmenter:
    def __init__(self):
        pass
    
    def threshold_segmentation(self, image, method='otsu', **kwargs):
        """
        阈值分割
        
        Args:
            image: 输入图像
            method: 阈值方法 (binary, adaptive, otsu)
            **kwargs: 参数
            
        Returns:
            分割后的二值图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'binary':
            thresh = kwargs.get('thresh', 127)
            maxval = kwargs.get('maxval', 255)
            _, binary = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
            return binary
        
        elif method == 'binary_inv':
            thresh = kwargs.get('thresh', 127)
            maxval = kwargs.get('maxval', 255)
            _, binary = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV)
            return binary
        
        elif method == 'adaptive':
            maxval = kwargs.get('maxval', 255)
            adaptive_method = kwargs.get('adaptive_method', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            threshold_type = kwargs.get('threshold_type', cv2.THRESH_BINARY)
            block_size = kwargs.get('block_size', 11)
            C = kwargs.get('C', 2)
            binary = cv2.adaptiveThreshold(gray, maxval, adaptive_method, threshold_type, block_size, C)
            return binary
        
        elif method == 'otsu':
            maxval = kwargs.get('maxval', 255)
            _, binary = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        else:
            raise ValueError(f"不支持的阈值方法: {method}")
    
    def watershed_segmentation(self, image, **kwargs):
        """
        分水岭分割
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            分割结果和标签
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        thresh_type = kwargs.get('thresh_type', 'otsu')
        binary = self.threshold_segmentation(gray, method=thresh_type, **kwargs)
        
        kernel = kwargs.get('kernel', np.ones((3, 3), np.uint8))
        opening_iterations = kwargs.get('opening_iterations', 2)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        dist_threshold = kwargs.get('dist_threshold', 0.7)
        _, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        
        markers = markers + 1
        markers[unknown == 255] = 0
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        
        return {
            'segmented_image': image,
            'markers': markers,
            'binary': binary,
            'sure_fg': sure_fg,
            'sure_bg': sure_bg
        }
    
    def grabcut_segmentation(self, image, rect=None, mask=None, **kwargs):
        """
        GrabCut分割
        
        Args:
            image: 输入图像
            rect: 矩形区域 (x, y, w, h)
            mask: 初始掩码
            **kwargs: 参数
            
        Returns:
            分割结果
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if mask is None:
            mask = np.zeros(image.shape[:2], np.uint8)
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        iter_count = kwargs.get('iter_count', 5)
        mode = kwargs.get('mode', cv2.GC_INIT_WITH_RECT if rect is not None else cv2.GC_INIT_WITH_MASK)
        
        if rect is not None:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iter_count, mode)
        else:
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, iter_count, mode)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = image * mask2[:, :, np.newaxis]
        
        return {
            'segmented_image': result,
            'mask': mask,
            'mask2': mask2
        }
    
    def region_growing(self, image, seed_points, **kwargs):
        """
        区域生长分割
        
        Args:
            image: 输入图像
            seed_points: 种子点列表 [(x1, y1), (x2, y2), ...]
            **kwargs: 参数
            
        Returns:
            分割结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        threshold = kwargs.get('threshold', 10)
        connectivity = kwargs.get('connectivity', 8)
        
        h, w = gray.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        
        for seed in seed_points:
            x, y = seed
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            
            seed_value = gray[y, x]
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            
            cv2.floodFill(gray, mask, (x, y), 255, 
                         loDiff=threshold, upDiff=threshold,
                         flags=connectivity | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
            
            segmented = np.maximum(segmented, mask[1:-1, 1:-1])
        
        return segmented
    
    def kmeans_segmentation(self, image, n_clusters=3, **kwargs):
        """
        K-means聚类分割
        
        Args:
            image: 输入图像
            n_clusters: 聚类数量
            **kwargs: 参数
            
        Returns:
            分割结果
        """
        if len(image.shape) == 3:
            pixels = image.reshape((-1, 3))
        else:
            pixels = image.reshape((-1, 1))
        
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = n_clusters
        
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        
        if len(image.shape) == 3:
            segmented = segmented.reshape(image.shape)
        else:
            segmented = segmented.reshape(image.shape[:2])
        
        return {
            'segmented_image': segmented,
            'labels': labels,
            'centers': centers
        }
    
    def mean_shift_segmentation(self, image, **kwargs):
        """
        Mean Shift分割
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            分割结果
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        spatial_radius = kwargs.get('spatial_radius', 10)
        color_radius = kwargs.get('color_radius', 10)
        max_level = kwargs.get('max_level', 1)
        
        segmented = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius, max_level)
        
        return segmented
    
    def contour_segmentation(self, image, **kwargs):
        """
        轮廓分割
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            轮廓和分割结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        thresh_type = kwargs.get('thresh_type', 'otsu')
        binary = self.threshold_segmentation(gray, method=thresh_type, **kwargs)
        
        mode = kwargs.get('mode', cv2.RETR_EXTERNAL)
        method = kwargs.get('method', cv2.CHAIN_APPROX_SIMPLE)
        
        contours, hierarchy = cv2.findContours(binary, mode, method)
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        min_area = kwargs.get('min_area', 0)
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered_contours.append(contour)
        
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)
        
        return {
            'contours': filtered_contours,
            'hierarchy': hierarchy,
            'segmented_image': result,
            'binary': binary
        }
    
    def edge_based_segmentation(self, image, **kwargs):
        """
        基于边缘的分割
        
        Args:
            image: 输入图像
            **kwargs: 参数
            
        Returns:
            分割结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edge_method = kwargs.get('edge_method', 'canny')
        
        if edge_method == 'canny':
            threshold1 = kwargs.get('threshold1', 100)
            threshold2 = kwargs.get('threshold2', 200)
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif edge_method == 'sobel':
            dx = kwargs.get('dx', 1)
            dy = kwargs.get('dy', 1)
            ksize = kwargs.get('ksize', 3)
            edges = cv2.Sobel(gray, cv2.CV_8U, dx, dy, ksize=ksize)
        else:
            raise ValueError(f"不支持的边缘检测方法: {edge_method}")
        
        kernel = kwargs.get('kernel', np.ones((3, 3), np.uint8))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return {
            'edges': edges,
            'edges_closed': edges_closed,
            'contours': contours,
            'segmented_image': result
        }
    
    def color_based_segmentation(self, image, lower_color, upper_color, color_space='bgr', **kwargs):
        """
        基于颜色的分割
        
        Args:
            image: 输入图像
            lower_color: 下限颜色
            upper_color: 上限颜色
            color_space: 颜色空间 (bgr, hsv, lab)
            **kwargs: 参数
            
        Returns:
            分割结果
        """
        if len(image.shape) != 3:
            raise ValueError("输入必须是彩色图像")
        
        if color_space == 'hsv':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'lab':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            converted = image
        
        mask = cv2.inRange(converted, np.array(lower_color), np.array(upper_color))
        
        kernel = kwargs.get('kernel', np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return {
            'mask': mask,
            'segmented_image': result
        }
    
    def draw_segmentation(self, image, segmentation_result, **kwargs):
        """
        绘制分割结果
        
        Args:
            image: 输入图像
            segmentation_result: 分割结果
            **kwargs: 参数
            
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        if 'markers' in segmentation_result:
            markers = segmentation_result['markers']
            result[markers == -1] = [255, 0, 0]
        
        if 'contours' in segmentation_result:
            color = kwargs.get('color', (0, 255, 0))
            thickness = kwargs.get('thickness', 2)
            cv2.drawContours(result, segmentation_result['contours'], -1, color, thickness)
        
        return result
