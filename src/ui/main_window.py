from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QTabWidget, 
                            QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, 
                            QCheckBox, QTextEdit, QSplitter)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from core.image_processing import ImageProcessor
from core.analysis import ImageAnalyzer
from deep_learning.model_trainer import ModelTrainer
from deep_learning.model_manager import ModelManager
from deployment.model_deployer import ModelDeployer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('工业机器视觉算法开发平台')
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化核心组件
        self.image_processor = ImageProcessor()
        self.image_analyzer = ImageAnalyzer()
        self.model_trainer = ModelTrainer()
        self.model_manager = ModelManager()
        self.model_deployer = ModelDeployer()
        
        # 当前图像
        self.current_image = None
        self.processed_image = None
        
        # 创建主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # 添加选项卡
        self.add_image_processing_tab()
        self.add_analysis_tab()
        self.add_deep_learning_tab()
        self.add_deployment_tab()
        
        # 状态栏
        self.statusBar().showMessage('就绪')
    
    def add_image_processing_tab(self):
        """
        添加图像处理选项卡
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 文件操作
        file_group = QGroupBox('文件操作')
        file_layout = QVBoxLayout(file_group)
        
        load_btn = QPushButton('加载图像')
        load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(load_btn)
        
        save_btn = QPushButton('保存图像')
        save_btn.clicked.connect(self.save_image)
        file_layout.addWidget(save_btn)
        
        control_layout.addWidget(file_group)
        
        # 滤波操作
        filter_group = QGroupBox('滤波操作')
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_type = QComboBox()
        self.filter_type.addItems(['gaussian', 'median', 'bilateral', 'average'])
        filter_layout.addWidget(QLabel('滤波类型:'))
        filter_layout.addWidget(self.filter_type)
        
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(1, 11)
        self.kernel_size.setValue(5)
        filter_layout.addWidget(QLabel(' kernel大小:'))
        filter_layout.addWidget(self.kernel_size)
        
        apply_filter_btn = QPushButton('应用滤波')
        apply_filter_btn.clicked.connect(self.apply_filter)
        filter_layout.addWidget(apply_filter_btn)
        
        control_layout.addWidget(filter_group)
        
        # 边缘检测
        edge_group = QGroupBox('边缘检测')
        edge_layout = QVBoxLayout(edge_group)
        
        self.edge_method = QComboBox()
        self.edge_method.addItems(['canny', 'sobel', 'laplacian'])
        edge_layout.addWidget(QLabel('边缘检测方法:'))
        edge_layout.addWidget(self.edge_method)
        
        apply_edge_btn = QPushButton('应用边缘检测')
        apply_edge_btn.clicked.connect(self.apply_edge_detection)
        edge_layout.addWidget(apply_edge_btn)
        
        control_layout.addWidget(edge_group)
        
        # 右侧图像显示
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        self.original_label = QLabel('原始图像')
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        image_layout.addWidget(self.original_label)
        
        self.processed_label = QLabel('处理后图像')
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(400, 300)
        image_layout.addWidget(self.processed_label)
        
        splitter.addWidget(control_panel)
        splitter.addWidget(image_panel)
        
        layout.addWidget(splitter)
        self.tab_widget.addTab(tab, '图像处理')
    
    def add_analysis_tab(self):
        """
        添加图像分析选项卡
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 目标检测
        detection_group = QGroupBox('目标检测')
        detection_layout = QVBoxLayout(detection_group)
        
        self.min_area = QSpinBox()
        self.min_area.setRange(10, 1000)
        self.min_area.setValue(100)
        detection_layout.addWidget(QLabel('最小面积:'))
        detection_layout.addWidget(self.min_area)
        
        detect_btn = QPushButton('检测目标')
        detect_btn.clicked.connect(self.detect_objects)
        detection_layout.addWidget(detect_btn)
        
        control_layout.addWidget(detection_group)
        
        # 缺陷检测
        defect_group = QGroupBox('缺陷检测')
        defect_layout = QVBoxLayout(defect_group)
        
        detect_defect_btn = QPushButton('检测缺陷')
        detect_defect_btn.clicked.connect(self.detect_defects)
        defect_layout.addWidget(detect_defect_btn)
        
        control_layout.addWidget(defect_group)
        
        # 右侧结果显示
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        
        self.result_label = QLabel('分析结果')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 300)
        result_layout.addWidget(self.result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(result_panel)
        
        layout.addWidget(splitter)
        self.tab_widget.addTab(tab, '图像分析')
    
    def add_deep_learning_tab(self):
        """
        添加深度学习选项卡
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型训练
        train_group = QGroupBox('模型训练')
        train_layout = QVBoxLayout(train_group)
        
        self.model_type = QComboBox()
        self.model_type.addItems(['classification', 'detection', 'segmentation'])
        train_layout.addWidget(QLabel('模型类型:'))
        train_layout.addWidget(self.model_type)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(10)
        train_layout.addWidget(QLabel('训练轮数:'))
        train_layout.addWidget(self.epochs)
        
        train_btn = QPushButton('开始训练')
        train_btn.clicked.connect(self.train_model)
        train_layout.addWidget(train_btn)
        
        layout.addWidget(train_group)
        
        # 模型管理
        manage_group = QGroupBox('模型管理')
        manage_layout = QVBoxLayout(manage_group)
        
        list_models_btn = QPushButton('列出模型')
        list_models_btn.clicked.connect(self.list_models)
        manage_layout.addWidget(list_models_btn)
        
        self.model_list = QTextEdit()
        self.model_list.setReadOnly(True)
        manage_layout.addWidget(self.model_list)
        
        layout.addWidget(manage_group)
        
        self.tab_widget.addTab(tab, '深度学习')
    
    def add_deployment_tab(self):
        """
        添加部署选项卡
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型导出
        export_group = QGroupBox('模型导出')
        export_layout = QVBoxLayout(export_group)
        
        export_btn = QPushButton('导出模型')
        export_btn.clicked.connect(self.export_model)
        export_layout.addWidget(export_btn)
        
        export_tflite_btn = QPushButton('导出为TFLite')
        export_tflite_btn.clicked.connect(self.export_tflite)
        export_layout.addWidget(export_tflite_btn)
        
        layout.addWidget(export_group)
        
        # 模型推理
        inference_group = QGroupBox('模型推理')
        inference_layout = QVBoxLayout(inference_group)
        
        load_model_btn = QPushButton('加载模型')
        load_model_btn.clicked.connect(self.load_model)
        inference_layout.addWidget(load_model_btn)
        
        infer_btn = QPushButton('推理')
        infer_btn.clicked.connect(self.infer_model)
        inference_layout.addWidget(infer_btn)
        
        self.inference_result = QTextEdit()
        self.inference_result.setReadOnly(True)
        inference_layout.addWidget(self.inference_result)
        
        layout.addWidget(inference_group)
        
        self.tab_widget.addTab(tab, '模型部署')
    
    def load_image(self):
        """
        加载图像
        """
        file_path, _ = QFileDialog.getOpenFileName(self, '加载图像', '', 'Image files (*.jpg *.jpeg *.png *.bmp)')
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.processed_image = self.current_image.copy()
            self.display_image(self.current_image, self.original_label)
            self.display_image(self.processed_image, self.processed_label)
            self.statusBar().showMessage(f'加载图像: {file_path}')
    
    def save_image(self):
        """
        保存图像
        """
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, '保存图像', '', 'Image files (*.jpg *.jpeg *.png *.bmp)')
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                self.statusBar().showMessage(f'保存图像: {file_path}')
    
    def apply_filter(self):
        """
        应用滤波
        """
        if self.current_image is not None:
            filter_type = self.filter_type.currentText()
            kernel_size = (self.kernel_size.value(), self.kernel_size.value())
            self.processed_image = self.image_processor.apply_filter(self.current_image, filter_type, kernel_size=kernel_size)
            self.display_image(self.processed_image, self.processed_label)
            self.statusBar().showMessage(f'应用{filter_type}滤波')
    
    def apply_edge_detection(self):
        """
        应用边缘检测
        """
        if self.current_image is not None:
            method = self.edge_method.currentText()
            edges = self.image_processor.detect_edges(self.current_image, method)
            # 转换为8位图像
            edges = cv2.convertScaleAbs(edges)
            self.processed_image = edges
            self.display_image(self.processed_image, self.processed_label)
            self.statusBar().showMessage(f'应用{method}边缘检测')
    
    def detect_objects(self):
        """
        检测目标
        """
        if self.current_image is not None:
            min_area = self.min_area.value()
            objects = self.image_analyzer.detect_objects(self.current_image, min_area=min_area)
            
            # 绘制检测结果
            result = self.current_image.copy()
            for obj in objects:
                x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            self.processed_image = result
            self.display_image(self.processed_image, self.result_label)
            
            # 显示结果信息
            result_text = f'检测到 {len(objects)} 个目标:\n'
            for i, obj in enumerate(objects):
                result_text += f'目标 {i+1}: 位置=({obj["x"]}, {obj["y"]}), 大小=({obj["width"]}, {obj["height"]}), 面积={obj["area"]}\n'
            
            self.result_text.setText(result_text)
            self.statusBar().showMessage(f'检测到 {len(objects)} 个目标')
    
    def detect_defects(self):
        """
        检测缺陷
        """
        if self.current_image is not None:
            defects = self.image_analyzer.detect_defects(self.current_image)
            
            # 绘制检测结果
            result = self.current_image.copy()
            for defect in defects:
                x, y, w, h = defect['x'], defect['y'], defect['width'], defect['height']
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            self.processed_image = result
            self.display_image(self.processed_image, self.result_label)
            
            # 显示结果信息
            result_text = f'检测到 {len(defects)} 个缺陷:\n'
            for i, defect in enumerate(defects):
                result_text += f'缺陷 {i+1}: 位置=({defect["x"]}, {defect["y"]}), 大小=({defect["width"]}, {defect["height"]}), 面积={defect["area"]}\n'
            
            self.result_text.setText(result_text)
            self.statusBar().showMessage(f'检测到 {len(defects)} 个缺陷')
    
    def train_model(self):
        """
        训练模型
        """
        # 这里简化处理，实际需要数据准备和模型训练
        self.statusBar().showMessage('开始训练模型...')
        # 模拟训练过程
        import time
        time.sleep(2)
        self.statusBar().showMessage('模型训练完成')
    
    def list_models(self):
        """
        列出模型
        """
        models = self.model_manager.list_models()
        model_text = '模型列表:\n\n'
        for model in models:
            model_text += f'ID: {model["id"]}\n'
            model_text += f'名称: {model["name"]}\n'
            model_text += f'类型: {model["type"]}\n'
            model_text += f'创建时间: {model["created_at"]}\n'
            model_text += '\n'
        
        self.model_list.setText(model_text)
    
    def export_model(self):
        """
        导出模型
        """
        self.statusBar().showMessage('导出模型...')
        # 模拟导出过程
        import time
        time.sleep(1)
        self.statusBar().showMessage('模型导出完成')
    
    def export_tflite(self):
        """
        导出为TFLite
        """
        self.statusBar().showMessage('导出为TFLite...')
        # 模拟导出过程
        import time
        time.sleep(1)
        self.statusBar().showMessage('TFLite导出完成')
    
    def load_model(self):
        """
        加载模型
        """
        file_path, _ = QFileDialog.getOpenFileName(self, '加载模型', '', 'Model files (*.h5 *.pb *.tflite)')
        if file_path:
            self.statusBar().showMessage(f'加载模型: {file_path}')
    
    def infer_model(self):
        """
        模型推理
        """
        if self.current_image is not None:
            self.statusBar().showMessage('执行推理...')
            # 模拟推理过程
            import time
            time.sleep(1)
            self.inference_result.setText('推理结果: 分类为 0')
            self.statusBar().showMessage('推理完成')
    
    def display_image(self, image, label):
        """
        显示图像
        """
        if image is not None:
            # 转换BGR为RGB
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 转换为QImage
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放图像以适应标签
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)