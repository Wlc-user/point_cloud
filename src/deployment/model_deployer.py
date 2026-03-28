import tensorflow as tf
import cv2
import numpy as np
import os

class ModelDeployer:
    def __init__(self):
        pass
    
    def export_model(self, model, export_dir, **kwargs):
        """
        导出模型
        
        Args:
            model: Keras模型
            export_dir: 导出目录
            **kwargs: 导出参数
        """
        # 导出为SavedModel格式
        model.save(export_dir, save_format='tf')
    
    def export_tflite(self, model, export_path, **kwargs):
        """
        导出为TFLite格式
        
        Args:
            model: Keras模型
            export_path: 导出路径
            **kwargs: 导出参数
        """
        # 转换为TFLite模型
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # 保存TFLite模型
        with open(export_path, 'wb') as f:
            f.write(tflite_model)
    
    def load_saved_model(self, model_path):
        """
        加载SavedModel格式模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的模型
        """
        return tf.saved_model.load(model_path)
    
    def load_tflite_model(self, model_path):
        """
        加载TFLite格式模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的TFLite解释器
        """
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def predict(self, model, image, **kwargs):
        """
        使用模型进行预测
        
        Args:
            model: 模型
            image: 输入图像
            **kwargs: 预测参数
            
        Returns:
            预测结果
        """
        # 预处理图像
        processed_image = self.preprocess_image(image, **kwargs)
        
        # 进行预测
        if isinstance(model, tf.lite.Interpreter):
            # 使用TFLite模型
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            model.set_tensor(input_details[0]['index'], processed_image)
            model.invoke()
            
            output = model.get_tensor(output_details[0]['index'])
        else:
            # 使用Keras模型
            output = model.predict(processed_image)
        
        return output
    
    def preprocess_image(self, image, **kwargs):
        """
        预处理图像
        
        Args:
            image: 输入图像
            **kwargs: 预处理参数
            
        Returns:
            预处理后的图像
        """
        target_size = kwargs.get('target_size', (224, 224))
        
        # 调整图像大小
        resized = cv2.resize(image, target_size)
        
        # 转换为RGB
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        
        # 归一化
        resized = resized / 255.0
        
        # 添加批次维度
        resized = np.expand_dims(resized, axis=0)
        
        return resized
    
    def postprocess_output(self, output, **kwargs):
        """
        后处理输出
        
        Args:
            output: 模型输出
            **kwargs: 后处理参数
            
        Returns:
            后处理结果
        """
        output_type = kwargs.get('output_type', 'classification')
        
        if output_type == 'classification':
            # 分类结果
            return np.argmax(output, axis=1)[0]
        elif output_type == 'detection':
            # 目标检测结果
            # 这里简化处理，实际需要根据模型输出格式进行解析
            return output
        elif output_type == 'segmentation':
            # 图像分割结果
            return np.argmax(output, axis=-1)[0]
        else:
            return output
    
    def batch_predict(self, model, images, **kwargs):
        """
        批量预测
        
        Args:
            model: 模型
            images: 图像列表
            **kwargs: 预测参数
            
        Returns:
            预测结果列表
        """
        results = []
        for image in images:
            result = self.predict(model, image, **kwargs)
            results.append(result)
        return results
    
    def benchmark_model(self, model, test_images, **kwargs):
        """
        基准测试模型性能
        
        Args:
            model: 模型
            test_images: 测试图像列表
            **kwargs: 测试参数
            
        Returns:
            性能指标
        """
        import time
        
        start_time = time.time()
        
        # 批量预测
        results = self.batch_predict(model, test_images, **kwargs)
        
        end_time = time.time()
        
        inference_time = end_time - start_time
        fps = len(test_images) / inference_time
        
        return {
            'inference_time': inference_time,
            'fps': fps,
            'batch_size': len(test_images)
        }