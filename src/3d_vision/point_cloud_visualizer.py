import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

class PointCloudVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def plot_point_cloud(self, points, colors=None, title='Point Cloud', 
                      point_size=1, alpha=0.5, show=True):
        """
        绘制3D点云
        
        Args:
            points: 点云数据 (N, 3)
            colors: 点的颜色 (N, 3) 或 (N,)
            title: 图像标题
            point_size: 点大小
            alpha: 透明度
            show: 是否显示图像
            
        Returns:
            图形对象
        """
        if len(points) == 0:
            print("点云为空")
            return None
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 提取坐标
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # 设置颜色
        if colors is None:
            # 使用高度作为颜色
            colors = z
        elif len(colors.shape) == 1:
            # 单通道颜色
            colors = colors
        else:
            # RGB颜色
            colors = colors
        
        # 绘制点云
        scatter = self.ax.scatter(x, y, z, c=colors, s=point_size, 
                                alpha=alpha, cmap='jet')
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        
        # 添加颜色条
        if len(colors.shape) == 1:
            plt.colorbar(scatter, ax=self.ax, label='Value')
        
        if show:
            plt.show()
        
        return self.fig
    
    def plot_normals(self, points, normals, length=0.1, title='Normals', show=True):
        """
        绘制点云和法向量
        
        Args:
            points: 点云数据 (N, 3)
            normals: 法向量 (N, 3)
            length: 法向量显示长度
            title: 图像标题
            show: 是否显示图像
            
        Returns:
            图形对象
        """
        if len(points) == 0 or len(normals) == 0:
            print("点云或法向量为空")
            return None
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='b', s=10, alpha=0.5)
        
        # 绘制法向量
        for i in range(len(points)):
            # 起点
            start = points[i]
            # 终点
            end = points[i] + normals[i] * length
            
            self.ax.plot([start[0], end[0]], 
                      [start[1], end[1]], 
                      [start[2], end[2]], 'r-', linewidth=0.5)
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        
        if show:
            plt.show()
        
        return self.fig
    
    def plot_clusters(self, points, labels, title='Clustering Result', 
                    point_size=10, alpha=0.6, show=True):
        """
        绘制聚类结果
        
        Args:
            points: 点云数据 (N, 3)
            labels: 聚类标签 (N,)
            title: 图像标题
            point_size: 点大小
            alpha: 透明度
            show: 是否显示图像
            
        Returns:
            图形对象
        """
        if len(points) == 0 or len(labels) == 0:
            print("点云或标签为空")
            return None
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        
        # 为每个聚类分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制每个聚类
        for i, label in enumerate(unique_labels):
            if label == -1:  # 噪声点
                color = 'black'
                label_name = 'Noise'
            else:
                color = colors[i]
                label_name = f'Cluster {label}'
            
            mask = labels == label
            self.ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                       c=color, s=point_size, alpha=alpha, 
                       label=label_name)
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        self.ax.legend()
        
        if show:
            plt.show()
        
        return self.fig
    
    def plot_plane(self, points, plane, title='Plane Segmentation', show=True):
        """
        绘制平面分割结果
        
        Args:
            points: 点云数据 (N, 3)
            plane: 平面方程 (a, b, c, d)
            title: 图像标题
            show: 是否显示图像
            
        Returns:
            图形对象
        """
        if len(points) == 0 or plane is None:
            print("点云为空或平面方程无效")
            return None
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='b', s=10, alpha=0.5, label='Points')
        
        # 创建平面网格
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                           np.linspace(y_min, y_max, 50))
        
        # 计算z坐标
        a, b, c, d = plane
        if abs(c) > 1e-6:
            zz = -(a * xx + b * yy + d) / c
        else:
            zz = np.zeros_like(xx)
        
        # 绘制平面
        self.ax.plot_surface(xx, yy, zz, alpha=0.3, color='r', label='Plane')
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        self.ax.legend()
        
        if show:
            plt.show()
        
        return self.fig
    
    def plot_bounding_box(self, points, bbox, title='Bounding Box', show=True):
        """
        绘制边界框
        
        Args:
            points: 点云数据 (N, 3)
            bbox: 边界框 {'min': [...], 'max': [...]}
            title: 图像标题
            show: 是否显示图像
            
        Returns:
            图形对象
        """
        if len(points) == 0 or bbox is None:
            print("点云为空或边界框无效")
            return None
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='b', s=10, alpha=0.5)
        
        # 获取边界框坐标
        min_coords = bbox['min']
        max_coords = bbox['max']
        
        # 绘制边界框
        self._draw_cube(min_coords, max_coords)
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        
        if show:
            plt.show()
        
        return self.fig
    
    def _draw_cube(self, min_coords, max_coords):
        """
        绘制立方体
        """
        # 定义立方体的8个顶点
        vertices = []
        for x in [min_coords[0], max_coords[0]]:
            for y in [min_coords[1], max_coords[1]]:
                for z in [min_coords[2], max_coords[2]]:
                    vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # 定义立方体的12条边
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7), (4, 5),
            (4, 6), (5, 7), (6, 7)
        ]
        
        # 绘制边
        for edge in edges:
            start = vertices[edge[0]]
            end = vertices[edge[1]]
            self.ax.plot([start[0], end[0]], 
                      [start[1], end[1]], 
                      [start[2], end[2]], 'r-', linewidth=2)
    
    def plot_registration(self, source_points, target_points, 
                      transformed_source=None, title='ICP Registration', show=True):
        """
        绘制配准结果
        
        Args:
            source_points: 源点云 (N, 3)
            target_points: 目标点云 (M, 3)
            transformed_source: 变换后的源点云
            title: 图像标题
            show: 是否显示图像
            
        Returns:
            图形对象
        """
        if len(source_points) == 0 or len(target_points) == 0:
            print("源点云或目标点云为空")
            return None
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制目标点云（蓝色）
        self.ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                   c='b', s=10, alpha=0.5, label='Target')
        
        # 绘制源点云（红色）
        self.ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2],
                   c='r', s=10, alpha=0.5, label='Source')
        
        # 绘制变换后的源点云（绿色）
        if transformed_source is not None:
            self.ax.scatter(transformed_source[:, 0], transformed_source[:, 1], 
                       transformed_source[:, 2],
                       c='g', s=10, alpha=0.5, label='Transformed')
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        self.ax.legend()
        
        if show:
            plt.show()
        
        return self.fig
    
    def plot_curvature(self, points, curvatures, title='Curvature', 
                     point_size=10, alpha=0.6, show=True):
        """
        绘制曲率
        
        Args:
            points: 点云数据 (N, 3)
            curvatures: 曲率值 (N,)
            title: 图像标题
            point_size: 点大小
            alpha: 透明度
            show: 是否显示图像
            
        Returns:
            图形对象
        """
        if len(points) == 0 or len(curvatures) == 0:
            print("点云或曲率为空")
            return None
        
        # 创建3D图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制点云，颜色表示曲率
        scatter = self.ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=curvatures, s=point_size, 
                               alpha=alpha, cmap='jet')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=self.ax, label='Curvature')
        
        # 设置标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        
        if show:
            plt.show()
        
        return self.fig
    
    def save_plot(self, filename, dpi=300):
        """
        保存当前图形
        
        Args:
            filename: 文件名
            dpi: 图像分辨率
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"图形已保存到: {filename}")
    
    def close(self):
        """
        关闭当前图形
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None