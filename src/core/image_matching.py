import cv2
import numpy as np
from .feature_extraction import FeatureExtractor

class ImageMatcher:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.templates = {}
    
    def template_matching(self, image, template, method='ccoeff_normed', **kwargs):
        """
        模板匹配
        
        Args:
            image: 输入图像
            template: 模板图像
            method: 匹配方法
            **kwargs: 参数
            
        Returns:
            匹配结果字典
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
        
        method_map = {
            'sqdiff': cv2.TM_SQDIFF,
            'sqdiff_normed': cv2.TM_SQDIFF_NORMED,
            'ccorr': cv2.TM_CCORR,
            'ccorr_normed': cv2.TM_CCORR_NORMED,
            'ccoeff': cv2.TM_CCOEFF,
            'ccoeff_normed': cv2.TM_CCOEFF_NORMED
        }
        
        if method not in method_map:
            raise ValueError(f"不支持的匹配方法: {method}")
        
        cv_method = method_map[method]
        
        result = cv2.matchTemplate(gray_image, gray_template, cv_method)
        
        threshold = kwargs.get('threshold', 0.8)
        max_matches = kwargs.get('max_matches', 10)
        
        matches = []
        
        if method in ['sqdiff', 'sqdiff_normed']:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            locations = np.where(result <= threshold)
        else:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            locations = np.where(result >= threshold)
        
        h, w = gray_template.shape
        
        for pt in zip(*locations[::-1]):
            matches.append({
                'top_left': pt,
                'bottom_right': (pt[0] + w, pt[1] + h),
                'center': (pt[0] + w // 2, pt[1] + h // 2),
                'score': result[pt[1], pt[0]],
                'width': w,
                'height': h
            })
        
        matches = sorted(matches, key=lambda x: x['score'], 
                        reverse=(method not in ['sqdiff', 'sqdiff_normed']))
        
        matches = matches[:max_matches]
        
        return {
            'matches': matches,
            'result_map': result,
            'best_match': matches[0] if matches else None,
            'best_score': max_val if method not in ['sqdiff', 'sqdiff_normed'] else min_val
        }
    
    def draw_matches(self, image, matches, **kwargs):
        """
        绘制匹配结果
        
        Args:
            image: 输入图像
            matches: 匹配结果列表
            **kwargs: 绘制参数
            
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        color = kwargs.get('color', (0, 255, 0))
        thickness = kwargs.get('thickness', 2)
        show_score = kwargs.get('show_score', True)
        
        for match in matches:
            top_left = match['top_left']
            bottom_right = match['bottom_right']
            
            cv2.rectangle(result, top_left, bottom_right, color, thickness)
            
            if show_score and 'score' in match:
                center = match['center']
                score_text = f"{match['score']:.2f}"
                cv2.putText(result, score_text, (center[0], center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def feature_matching(self, image1, image2, method='orb', matcher='bf', **kwargs):
        """
        特征点匹配
        
        Args:
            image1: 图像1
            image2: 图像2
            method: 特征方法 (orb, sift, surf)
            matcher: 匹配器 (bf, flann)
            **kwargs: 参数
            
        Returns:
            匹配结果
        """
        kp1, des1 = self.feature_extractor.detect_and_compute(image1, method=method, **kwargs)
        kp2, des2 = self.feature_extractor.detect_and_compute(image2, method=method, **kwargs)
        
        if des1 is None or des2 is None:
            return {
                'keypoints1': kp1,
                'keypoints2': kp2,
                'matches': [],
                'good_matches': []
            }
        
        if matcher == 'bf':
            cross_check = kwargs.get('cross_check', True)
            
            if method == 'orb':
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
            else:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
            
            matches = bf.match(des1, des2)
            
        elif matcher == 'flann':
            if method == 'orb':
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            good_matches = []
            ratio_threshold = kwargs.get('ratio_threshold', 0.7)
            
            for m, n in matches:
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            
            matches = good_matches
        
        else:
            raise ValueError(f"不支持的匹配器: {matcher}")
        
        matches = sorted(matches, key=lambda x: x.distance)
        
        max_matches = kwargs.get('max_matches', 100)
        good_matches = matches[:max_matches]
        
        return {
            'keypoints1': kp1,
            'keypoints2': kp2,
            'matches': matches,
            'good_matches': good_matches
        }
    
    def draw_feature_matches(self, image1, image2, kp1, kp2, matches, **kwargs):
        """
        绘制特征匹配结果
        
        Args:
            image1: 图像1
            image2: 图像2
            kp1: 关键点1
            kp2: 关键点2
            matches: 匹配列表
            **kwargs: 参数
            
        Returns:
            绘制后的图像
        """
        match_color = kwargs.get('match_color', (0, 255, 0))
        single_point_color = kwargs.get('single_point_color', (255, 0, 0))
        flags = kwargs.get('flags', cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        result = cv2.drawMatches(image1, kp1, image2, kp2, matches, None,
                                matchColor=match_color,
                                singlePointColor=single_point_color,
                                flags=flags)
        
        return result
    
    def find_homography(self, image1, image2, method='orb', **kwargs):
        """
        计算单应性矩阵
        
        Args:
            image1: 图像1
            image2: 图像2
            method: 特征方法
            **kwargs: 参数
            
        Returns:
            单应性矩阵和结果
        """
        match_result = self.feature_matching(image1, image2, method=method, **kwargs)
        
        kp1 = match_result['keypoints1']
        kp2 = match_result['keypoints2']
        good_matches = match_result['good_matches']
        
        if len(good_matches) < 4:
            return {
                'homography': None,
                'inliers': [],
                'match_result': match_result
            }
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        ransac_threshold = kwargs.get('ransac_threshold', 5.0)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        
        inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
        
        return {
            'homography': M,
            'inliers': inliers,
            'mask': mask,
            'match_result': match_result
        }
    
    def stitch_images(self, image1, image2, method='orb', **kwargs):
        """
        图像拼接
        
        Args:
            image1: 图像1
            image2: 图像2
            method: 特征方法
            **kwargs: 参数
            
        Returns:
            拼接后的图像
        """
        homography_result = self.find_homography(image1, image2, method=method, **kwargs)
        M = homography_result['homography']
        
        if M is None:
            return None
        
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        result = cv2.warpPerspective(image1, M, (w1 + w2, max(h1, h2)))
        result[0:h2, 0:w2] = image2
        
        return result
    
    def add_template(self, name, template):
        """
        添加模板
        
        Args:
            name: 模板名称
            template: 模板图像
        """
        self.templates[name] = template
    
    def match_templates(self, image, template_names=None, **kwargs):
        """
        匹配多个模板
        
        Args:
            image: 输入图像
            template_names: 模板名称列表
            **kwargs: 参数
            
        Returns:
            匹配结果字典
        """
        if template_names is None:
            template_names = list(self.templates.keys())
        
        results = {}
        
        for name in template_names:
            if name in self.templates:
                results[name] = self.template_matching(image, self.templates[name], **kwargs)
        
        return results
    
    def object_tracking(self, video_path, template, **kwargs):
        """
        目标追踪
        
        Args:
            video_path: 视频路径
            template: 模板图像
            **kwargs: 参数
            
        Returns:
            追踪结果
        """
        tracker_type = kwargs.get('tracker_type', 'kcf')
        
        tracker_map = {
            'kcf': cv2.TrackerKCF_create,
            'csrt': cv2.TrackerCSRT_create,
            'mil': cv2.TrackerMIL_create,
            'boosting': cv2.TrackerBoosting_create,
            'medianflow': cv2.TrackerMedianFlow_create,
            'mosse': cv2.TrackerMOSSE_create
        }
        
        if tracker_type not in tracker_map:
            raise ValueError(f"不支持的追踪器类型: {tracker_type}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
        
        match_result = self.template_matching(frame, template, **kwargs)
        
        if not match_result['matches']:
            cap.release()
            return None
        
        best_match = match_result['matches'][0]
        bbox = (best_match['top_left'][0], best_match['top_left'][1],
               best_match['width'], best_match['height'])
        
        tracker = tracker_map[tracker_type]()
        tracker.init(frame, bbox)
        
        track_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            success, bbox = tracker.update(frame)
            
            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                track_results.append({
                    'frame': frame.copy(),
                    'bbox': bbox,
                    'success': True
                })
            else:
                track_results.append({
                    'frame': frame.copy(),
                    'bbox': None,
                    'success': False
                })
        
        cap.release()
        
        return track_results
