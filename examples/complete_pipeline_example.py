import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src_core.image_capture import ImageCapture
from src_core.image_processing import ImageProcessor
from src_core.image_segmentation import ImageSegmenter
from src_core.feature_extraction import FeatureExtractor
from src_core.image_matching import ImageMatcher
from src_core.measurement import PrecisionMeasurement
from src_3d_vision.point_cloud import PointCloudProcessor

class CompleteVisionPipeline:
    def __init__(self):
        self.capture = ImageCapture()
        self.processor = ImageProcessor()
        self.segmenter = ImageSegmenter()
        self.feature_extractor = FeatureExtractor()
        self.matcher = ImageMatcher()
        self.measurement = PrecisionMeasurement()
        self.point_cloud = PointCloudProcessor()
        
        self.template = None
        self.calibration_done = False
    
    def step1_capture_image(self, source='file', filepath=None, camera_id=0):
        """
        步骤1：图像采集
        
        Args:
            source: 'file' 或 'camera'
            filepath: 图像文件路径
            camera_id: 相机ID
            
        Returns:
            采集的图像
        """
        print("\n=== 步骤1：图像采集 ===")
        
        if source == 'file' and filepath:
            image = self.capture.load_image(filepath)
            if image is not None:
                print(f"从文件加载图像成功: {filepath}")
                return image
            else:
                print(f"无法加载图像: {filepath}")
                return None
        
        elif source == 'camera':
            print("尝试打开相机...")
            if self.capture.open_camera(camera_id):
                print(f"相机打开成功: ID={camera_id}")
                
                resolution = self.capture.get_resolution()
                fps = self.capture.get_fps()
                print(f"分辨率: {resolution}")
                print(f"帧率: {fps}")
                
                print("正在采集图像...")
                image = self.capture.capture_frame()
                
                self.capture.close_camera()
                
                if image is not None:
                    print("图像采集成功")
                    return image
                else:
                    print("图像采集失败")
                    return None
            else:
                print("相机打开失败")
                return None
        
        return None
    
    def step2_preprocess(self, image, **kwargs):
        """
        步骤2：图像预处理
        
        Args:
            image: 输入图像
            **kwargs: 预处理参数
            
        Returns:
            预处理后的图像
        """
        print("\n=== 步骤2：图像预处理 ===")
        
        result = image.copy()
        
        preprocess_steps = kwargs.get('steps', ['denoise', 'enhance'])
        
        for step in preprocess_steps:
            if step == 'denoise':
                filter_type = kwargs.get('filter_type', 'bilateral')
                print(f"应用去噪: {filter_type}")
                result = self.processor.apply_filter(result, filter_type, **kwargs)
            
            elif step == 'enhance':
                print("增强对比度...")
                if len(result.shape) == 3:
                    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    lab = cv2.merge([l, a, b])
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    result = clahe.apply(result)
            
            elif step == 'edges':
                print("边缘检测...")
                result = self.processor.detect_edges(result, method='canny', **kwargs)
        
        print("预处理完成")
        return result
    
    def step3_segment(self, image, method='contour', **kwargs):
        """
        步骤3：图像分割
        
        Args:
            image: 输入图像
            method: 分割方法
            **kwargs: 分割参数
            
        Returns:
            分割结果
        """
        print(f"\n=== 步骤3：图像分割 ({method}) ===")
        
        if method == 'threshold':
            result = self.segmenter.threshold_segmentation(image, **kwargs)
            print("阈值分割完成")
            return {'binary': result}
        
        elif method == 'contour':
            result = self.segmenter.contour_segmentation(image, **kwargs)
            print(f"轮廓分割完成，检测到 {len(result['contours'])} 个轮廓")
            return result
        
        elif method == 'watershed':
            result = self.segmenter.watershed_segmentation(image, **kwargs)
            print("分水岭分割完成")
            return result
        
        elif method == 'kmeans':
            result = self.segmenter.kmeans_segmentation(image, **kwargs)
            print(f"K-means分割完成，{kwargs.get('n_clusters', 3)} 个聚类")
            return result
        
        else:
            print(f"不支持的分割方法: {method}")
            return None
    
    def step4_extract_features(self, image, contours=None, **kwargs):
        """
        步骤4：特征提取
        
        Args:
            image: 输入图像
            contours: 轮廓列表
            **kwargs: 参数
            
        Returns:
            特征字典
        """
        print("\n=== 步骤4：特征提取 ===")
        
        features = {}
        
        if kwargs.get('keypoints', True):
            method = kwargs.get('keypoint_method', 'orb')
            print(f"提取关键点: {method}")
            keypoints, descriptors = self.feature_extractor.detect_and_compute(image, method=method)
            features['keypoints'] = keypoints
            features['descriptors'] = descriptors
            print(f"检测到 {len(keypoints)} 个关键点")
        
        if kwargs.get('color', True):
            print("提取颜色特征...")
            if len(image.shape) == 3:
                features['color'] = self.feature_extractor.extract_color_features(image)
        
        if kwargs.get('texture', True):
            print("提取纹理特征...")
            features['lbp'], _ = self.feature_extractor.extract_lbp_features(image)
            features['glcm'] = self.feature_extractor.extract_glcm_features(image)
        
        if kwargs.get('shape', True) and contours:
            print("提取形状特征...")
            shape_features = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > kwargs.get('min_area', 100):
                    sf = self.feature_extractor.extract_shape_features(contour)
                    shape_features.append(sf)
            features['shape'] = shape_features
            print(f"提取了 {len(shape_features)} 个形状特征")
        
        print("特征提取完成")
        return features
    
    def step5_match(self, image, template=None, method='template', **kwargs):
        """
        步骤5：图像匹配
        
        Args:
            image: 输入图像
            template: 模板图像
            method: 匹配方法
            **kwargs: 参数
            
        Returns:
            匹配结果
        """
        print(f"\n=== 步骤5：图像匹配 ({method}) ===")
        
        if method == 'template':
            if template is None and self.template is None:
                print("需要提供模板图像")
                return None
            
            use_template = template if template is not None else self.template
            
            result = self.matcher.template_matching(image, use_template, **kwargs)
            print(f"找到 {len(result['matches'])} 个匹配")
            if result['best_match']:
                print(f"最佳匹配分数: {result['best_score']:.4f}")
            return result
        
        elif method == 'feature':
            if template is None and self.template is None:
                print("需要提供模板图像")
                return None
            
            use_template = template if template is not None else self.template
            
            result = self.matcher.feature_matching(image, use_template, **kwargs)
            print(f"找到 {len(result['good_matches'])} 个良好匹配")
            return result
        
        else:
            print(f"不支持的匹配方法: {method}")
            return None
    
    def step6_measure(self, image, segmentation_result, **kwargs):
        """
        步骤6：测量
        
        Args:
            image: 输入图像
            segmentation_result: 分割结果
            **kwargs: 参数
            
        Returns:
            测量结果
        """
        print("\n=== 步骤6：测量 ===")
        
        if not self.calibration_done:
            print("未校准，使用像素单位")
            self.measurement.set_calibration(1.0, 'pixel')
        else:
            print(f"使用已校准的单位: {self.measurement.unit}")
        
        measurements = []
        
        if 'contours' in segmentation_result:
            contours = segmentation_result['contours']
            min_area = kwargs.get('min_area', 100)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= min_area:
                    meas = self.measurement.measure_contour_dimensions(contour)
                    meas['id'] = i
                    measurements.append(meas)
                    
                    print(f"\n对象 {i}:")
                    print(f"  面积: {meas['area']:.4f} {meas['unit']}")
                    print(f"  周长: {meas['perimeter']:.4f} {meas['unit']}")
                    print(f"  尺寸: {meas['bounding_box']['width']:.4f} x {meas['bounding_box']['height']:.4f} {meas['unit']}")
        
        if kwargs.get('holes', False):
            holes = self.measurement.measure_hole_position(image, **kwargs)
            print(f"\n检测到 {len(holes)} 个孔")
            for i, hole in enumerate(holes):
                print(f"  孔 {i}: 位置={hole['center_real']}, 直径={hole['diameter_real']:.4f} {hole['unit']}")
            measurements.append({'holes': holes})
        
        print(f"\n测量完成，共测量 {len(measurements)} 个对象")
        return measurements
    
    def step7_3d_processing(self, points, **kwargs):
        """
        步骤7：3D点云处理
        
        Args:
            points: 点云数据
            **kwargs: 参数
            
        Returns:
            点云处理结果
        """
        print("\n=== 步骤7：3D点云处理 ===")
        
        result = {}
        
        if kwargs.get('filter', True):
            print("点云滤波...")
            filtered, _ = self.point_cloud.statistical_outlier_removal(points)
            result['filtered'] = filtered
            print(f"滤波后: {len(filtered)} 个点")
        
        if kwargs.get('normals', True):
            print("计算法向量...")
            normals = self.point_cloud.compute_normals(filtered if 'filtered' in result else points)
            result['normals'] = normals
        
        if kwargs.get('segment', True):
            print("平面分割...")
            plane, inliers = self.point_cloud.plane_segmentation(filtered if 'filtered' in result else points)
            if plane is not None:
                result['plane'] = plane
                result['plane_inliers'] = inliers
                print(f"平面方程: {plane[0]:.3f}x + {plane[1]:.3f}y + {plane[2]:.3f}z + {plane[3]:.3f} = 0")
        
        if kwargs.get('cluster', True):
            print("点云聚类...")
            labels = self.point_cloud.euclidean_clustering(filtered if 'filtered' in result else points)
            result['labels'] = labels
            n_clusters = len(np.unique(labels[labels >= 0]))
            print(f"检测到 {n_clusters} 个聚类")
        
        print("3D点云处理完成")
        return result
    
    def calibrate(self, calibration_image, reference_length, **kwargs):
        """
        系统校准
        
        Args:
            calibration_image: 校准图像
            reference_length: 参考长度
            **kwargs: 参数
        """
        print("\n=== 系统校准 ===")
        success = self.measurement.calibrate_with_reference(
            calibration_image, reference_length, **kwargs
        )
        
        if success:
            self.calibration_done = True
            print(f"校准成功！1像素 = {self.measurement.pixel_to_unit:.6f} {self.measurement.unit}")
        else:
            print("校准失败")
        
        return success
    
    def set_template(self, template_image):
        """
        设置匹配模板
        
        Args:
            template_image: 模板图像
        """
        self.template = template_image
        print("模板已设置")
    
    def run_complete_pipeline(self, image_source='file', image_path=None, 
                            template_path=None, calibration_image=None, 
                            reference_length=None, **kwargs):
        """
        运行完整流程
        
        Args:
            image_source: 图像源 ('file' 或 'camera')
            image_path: 图像文件路径
            template_path: 模板图像路径
            calibration_image: 校准图像
            reference_length: 参考长度
            **kwargs: 其他参数
            
        Returns:
            完整结果字典
        """
        print("=" * 60)
        print("工业机器视觉完整处理流程")
        print("=" * 60)
        
        results = {}
        
        if calibration_image is not None and reference_length is not None:
            self.calibrate(calibration_image, reference_length, **kwargs)
        
        if template_path is not None:
            template = self.capture.load_image(template_path)
            if template is not None:
                self.set_template(template)
        
        image = self.step1_capture_image(image_source, image_path, **kwargs)
        if image is None:
            print("流程终止：无法获取图像")
            return None
        
        results['original'] = image
        
        preprocessed = self.step2_preprocess(image, **kwargs)
        results['preprocessed'] = preprocessed
        
        seg_result = self.step3_segment(preprocessed, **kwargs)
        results['segmentation'] = seg_result
        
        contours = seg_result.get('contours', []) if seg_result else []
        features = self.step4_extract_features(image, contours, **kwargs)
        results['features'] = features
        
        if self.template is not None:
            match_result = self.step5_match(image, **kwargs)
            results['matching'] = match_result
        
        if seg_result:
            measurements = self.step6_measure(image, seg_result, **kwargs)
            results['measurements'] = measurements
        
        print("\n" + "=" * 60)
        print("流程执行完成！")
        print("=" * 60)
        
        return results
    
    def save_results(self, results, output_dir='output'):
        """
        保存结果
        
        Args:
            results: 结果字典
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if 'original' in results:
            cv2.imwrite(os.path.join(output_dir, '01_original.jpg'), results['original'])
        
        if 'preprocessed' in results:
            cv2.imwrite(os.path.join(output_dir, '02_preprocessed.jpg'), results['preprocessed'])
        
        if 'segmentation' in results:
            seg = results['segmentation']
            if 'segmented_image' in seg:
                cv2.imwrite(os.path.join(output_dir, '03_segmentation.jpg'), seg['segmented_image'])
        
        print(f"结果已保存到: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='完整机器视觉处理流程演示')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--template', type=str, help='模板图像路径')
    parser.add_argument('--calibration', type=str, help='校准图像路径')
    parser.add_argument('--ref_length', type=float, help='参考长度')
    parser.add_argument('--camera', type=int, default=0, help='相机ID')
    parser.add_argument('--use_camera', action='store_true', help='使用相机')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    pipeline = CompleteVisionPipeline()
    
    calibration_image = None
    if args.calibration:
        calibration_image = cv2.imread(args.calibration)
    
    image_source = 'camera' if args.use_camera else 'file'
    image_path = None if args.use_camera else args.image
    
    results = pipeline.run_complete_pipeline(
        image_source=image_source,
        image_path=image_path,
        template_path=args.template,
        calibration_image=calibration_image,
        reference_length=args.ref_length,
        steps=['denoise', 'enhance'],
        filter_type='bilateral',
        method='contour',
        min_area=100
    )
    
    if results:
        pipeline.save_results(results, args.output)


if __name__ == '__main__':
    main()
