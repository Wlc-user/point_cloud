import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from point_cloud import PointCloudProcessor
from point_cloud_visualizer import PointCloudVisualizer

def generate_sample_point_cloud(n_points=1000, noise_level=0.01):
    """
    生成示例点云
    
    Args:
        n_points: 点数量
        noise_level: 噪声水平
        
    Returns:
        点云数据
    """
    # 生成平面点云
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = 0.5 * x + 0.3 * y + np.random.normal(0, noise_level, n_points)
    
    # 添加一些离群点
    n_outliers = int(n_points * 0.05)
    outlier_x = np.random.uniform(-1, 1, n_outliers)
    outlier_y = np.random.uniform(-1, 1, n_outliers)
    outlier_z = np.random.uniform(2, 3, n_outliers)
    
    # 合并点云
    points = np.column_stack([x, y, z])
    outliers = np.column_stack([outlier_x, outlier_y, outlier_z])
    all_points = np.vstack([points, outliers])
    
    return all_points

def generate_sphere_point_cloud(n_points=1000, radius=1.0, noise_level=0.01):
    """
    生成球体点云
    
    Args:
        n_points: 点数量
        radius: 半径
        noise_level: 噪声水平
        
    Returns:
        点云数据
    """
    # 使用球坐标生成点
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    # 添加噪声
    x += np.random.normal(0, noise_level, n_points)
    y += np.random.normal(0, noise_level, n_points)
    z += np.random.normal(0, noise_level, n_points)
    
    return np.column_stack([x, y, z])

def demo_point_cloud_filtering():
    """
    点云滤波演示
    """
    print("\n点云滤波演示")
    print("=" * 50)
    
    # 生成示例点云
    points = generate_sample_point_cloud(n_points=1000, noise_level=0.02)
    print(f"原始点云: {len(points)} 个点")
    
    # 创建处理器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # 1. 统计离群点移除
    print("\n1. 统计离群点移除...")
    filtered_points, inlier_indices = processor.statistical_outlier_removal(
        points, nb_neighbors=20, std_ratio=2.0)
    print(f"   过滤后: {len(filtered_points)} 个点")
    print(f"   移除: {len(points) - len(filtered_points)} 个离群点")
    
    # 2. 半径离群点移除
    print("\n2. 半径离群点移除...")
    filtered_points2, inlier_indices2 = processor.radius_outlier_removal(
        points, radius=0.1, min_neighbors=5)
    print(f"   过滤后: {len(filtered_points2)} 个点")
    print(f"   移除: {len(points) - len(filtered_points2)} 个离群点")
    
    # 3. 体素网格滤波
    print("\n3. 体素网格滤波...")
    downsampled_points = processor.voxel_grid_filter(points, voxel_size=0.05)
    print(f"   下采样后: {len(downsampled_points)} 个点")
    print(f"   压缩率: {(1 - len(downsampled_points)/len(points))*100:.1f}%")
    
    # 4. 双边滤波
    print("\n4. 双边滤波...")
    bilateral_points = processor.bilateral_filter(
        points, sigma_spatial=0.1, sigma_range=0.1, k=20)
    print(f"   滤波完成")
    
    # 可视化
    print("\n可视化结果...")
    visualizer.plot_point_cloud(points, title='Original Point Cloud', show=False)
    visualizer.save_plot('original_point_cloud.png')
    visualizer.close()
    
    visualizer.plot_point_cloud(filtered_points, title='Statistical Filtered', show=False)
    visualizer.save_plot('statistical_filtered.png')
    visualizer.close()
    
    visualizer.plot_point_cloud(downsampled_points, title='Voxel Downsampled', show=False)
    visualizer.save_plot('voxel_downsampled.png')
    visualizer.close()

def demo_normal_estimation():
    """
    法向量估计演示
    """
    print("\n法向量估计演示")
    print("=" * 50)
    
    # 生成球体点云
    points = generate_sphere_point_cloud(n_points=500, radius=1.0)
    print(f"点云: {len(points)} 个点")
    
    # 创建处理器和可视化器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # 计算法向量
    print("\n计算法向量...")
    normals = processor.compute_normals(points, k=10)
    print(f"法向量计算完成")
    
    # 可视化
    visualizer.plot_normals(points, normals, length=0.2, 
                        title='Point Cloud with Normals')
    visualizer.save_plot('normals.png')
    visualizer.close()

def demo_icp_registration():
    """
    ICP配准演示
    """
    print("\nICP配准演示")
    print("=" * 50)
    
    # 生成源点云和目标点云
    source_points = generate_sample_point_cloud(n_points=500, noise_level=0.01)
    
    # 对目标点云进行变换
    rotation_angle = np.pi / 6  # 30度
    translation = np.array([0.2, 0.1, 0.3])
    
    # 旋转矩阵
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # 应用变换
    target_points = np.dot(source_points, rotation_matrix.T) + translation
    
    print(f"源点云: {len(source_points)} 个点")
    print(f"目标点云: {len(target_points)} 个点")
    print(f"变换: 旋转{rotation_angle*180/np.pi:.1f}°, 平移{translation}")
    
    # 创建处理器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # ICP配准
    print("\n执行ICP配准...")
    transformation, aligned_source = processor.icp_registration(
        source_points, target_points, 
        max_iterations=50, 
        tolerance=1e-6,
        max_distance=0.1
    )
    
    print(f"配准完成")
    print(f"变换矩阵:\n{transformation}")
    
    # 计算配准误差
    error = np.mean(np.linalg.norm(aligned_source - target_points, axis=1))
    print(f"平均配准误差: {error:.4f}")
    
    # 可视化
    visualizer.plot_registration(source_points, target_points, 
                            aligned_source, title='ICP Registration')
    visualizer.save_plot('icp_registration.png')
    visualizer.close()

def demo_clustering():
    """
    点云聚类演示
    """
    print("\n点云聚类演示")
    print("=" * 50)
    
    # 生成多个聚类的点云
    cluster1 = generate_sample_point_cloud(n_points=300, noise_level=0.01)
    cluster2 = generate_sample_point_cloud(n_points=300, noise_level=0.01)
    
    # 对第二个聚类进行平移
    cluster2[:, 0] += 2.0
    cluster2[:, 1] += 1.0
    
    # 合并点云
    points = np.vstack([cluster1, cluster2])
    print(f"总点云: {len(points)} 个点")
    
    # 创建处理器和可视化器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # 1. 欧几里得聚类
    print("\n1. 欧几里得聚类...")
    labels = processor.euclidean_clustering(
        points, tolerance=0.1, min_cluster_size=50)
    n_clusters = len(np.unique(labels[labels >= 0]))
    print(f"   检测到 {n_clusters} 个聚类")
    
    # 2. K-means聚类
    print("\n2. K-means聚类...")
    labels_kmeans, centers = processor.kmeans_clustering(points, n_clusters=2)
    print(f"   聚类中心:\n{centers}")
    
    # 可视化
    visualizer.plot_clusters(points, labels, title='Euclidean Clustering')
    visualizer.save_plot('euclidean_clustering.png')
    visualizer.close()
    
    visualizer.plot_clusters(points, labels_kmeans, title='K-means Clustering')
    visualizer.save_plot('kmeans_clustering.png')
    visualizer.close()

def demo_plane_segmentation():
    """
    平面分割演示
    """
    print("\n平面分割演示")
    print("=" * 50)
    
    # 生成平面点云
    points = generate_sample_point_cloud(n_points=800, noise_level=0.01)
    print(f"点云: {len(points)} 个点")
    
    # 创建处理器和可视化器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # 平面分割
    print("\n执行平面分割...")
    plane, inliers = processor.plane_segmentation(
        points, distance_threshold=0.05, 
        ransac_n=3, 
        num_iterations=1000
    )
    
    if plane is not None:
        print(f"平面方程: {plane[0]:.3f}x + {plane[1]:.3f}y + {plane[2]:.3f}z + {plane[3]:.3f} = 0")
        print(f"内点数量: {len(inliers)}")
        print(f"内点比例: {len(inliers)/len(points)*100:.1f}%")
    else:
        print("平面分割失败")
    
    # 可视化
    visualizer.plot_plane(points, plane, title='Plane Segmentation')
    visualizer.save_plot('plane_segmentation.png')
    visualizer.close()

def demo_curvature_estimation():
    """
    曲率估计演示
    """
    print("\n曲率估计演示")
    print("=" * 50)
    
    # 生成球体点云
    points = generate_sphere_point_cloud(n_points=500, radius=1.0)
    print(f"点云: {len(points)} 个点")
    
    # 创建处理器和可视化器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # 计算曲率
    print("\n计算曲率...")
    curvatures = processor.compute_curvature(points, k=10)
    print(f"曲率计算完成")
    print(f"平均曲率: {np.mean(curvatures):.4f}")
    print(f"最大曲率: {np.max(curvatures):.4f}")
    print(f"最小曲率: {np.min(curvatures):.4f}")
    
    # 可视化
    visualizer.plot_curvature(points, curvatures, title='Curvature Estimation')
    visualizer.save_plot('curvature.png')
    visualizer.close()

def demo_bounding_box():
    """
    边界框演示
    """
    print("\n边界框演示")
    print("=" * 50)
    
    # 生成示例点云
    points = generate_sample_point_cloud(n_points=1000, noise_level=0.01)
    print(f"点云: {len(points)} 个点")
    
    # 创建处理器和可视化器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # 计算边界框
    print("\n计算边界框...")
    bbox = processor.bounding_box(points)
    
    if bbox is not None:
        print(f"最小坐标: {bbox['min']}")
        print(f"最大坐标: {bbox['max']}")
        print(f"尺寸: {bbox['size']}")
        print(f"中心: {bbox['center']}")
    
    # 可视化
    visualizer.plot_bounding_box(points, bbox, title='Bounding Box')
    visualizer.save_plot('bounding_box.png')
    visualizer.close()

def demo_comprehensive_pipeline():
    """
    综合处理流程演示
    """
    print("\n综合点云处理流程演示")
    print("=" * 50)
    
    # 生成示例点云
    points = generate_sample_point_cloud(n_points=2000, noise_level=0.02)
    print(f"原始点云: {len(points)} 个点")
    
    # 创建处理器和可视化器
    processor = PointCloudProcessor()
    visualizer = PointCloudVisualizer()
    
    # 处理流程
    print("\n处理流程:")
    
    # 1. 离群点移除
    print("1. 离群点移除...")
    filtered_points, _ = processor.statistical_outlier_removal(points)
    print(f"   剩余: {len(filtered_points)} 个点")
    
    # 2. 下采样
    print("2. 下采样...")
    downsampled_points = processor.voxel_grid_filter(filtered_points, voxel_size=0.05)
    print(f"   下采样: {len(downsampled_points)} 个点")
    
    # 3. 计算法向量
    print("3. 计算法向量...")
    normals = processor.compute_normals(downsampled_points)
    print(f"   法向量计算完成")
    
    # 4. 平面分割
    print("4. 平面分割...")
    plane, inliers = processor.plane_segmentation(downsampled_points)
    if plane is not None:
        print(f"   检测到平面")
    
    # 5. 聚类
    print("5. 聚类...")
    labels = processor.euclidean_clustering(downsampled_points)
    n_clusters = len(np.unique(labels[labels >= 0]))
    print(f"   检测到 {n_clusters} 个聚类")
    
    # 6. 计算边界框
    print("6. 计算边界框...")
    bbox = processor.bounding_box(downsampled_points)
    if bbox is not None:
        print(f"   边界框: {bbox['size']}")
    
    # 可视化结果
    print("\n可视化结果...")
    visualizer.plot_point_cloud(points, title='Original', show=False)
    visualizer.save_plot('pipeline_original.png')
    visualizer.close()
    
    visualizer.plot_point_cloud(downsampled_points, title='Processed', show=False)
    visualizer.save_plot('pipeline_processed.png')
    visualizer.close()
    
    visualizer.plot_clusters(downsampled_points, labels, title='Clustering', show=False)
    visualizer.save_plot('pipeline_clustering.png')
    visualizer.close()
    
    print("\n处理完成!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='点云处理演示')
    parser.add_argument('--mode', type=str, default='filtering',
                       choices=['filtering', 'normals', 'icp', 
                               'clustering', 'plane', 'curvature', 
                               'bbox', 'pipeline'],
                       help='演示模式')
    
    args = parser.parse_args()
    
    if args.mode == 'filtering':
        demo_point_cloud_filtering()
    elif args.mode == 'normals':
        demo_normal_estimation()
    elif args.mode == 'icp':
        demo_icp_registration()
    elif args.mode == 'clustering':
        demo_clustering()
    elif args.mode == 'plane':
        demo_plane_segmentation()
    elif args.mode == 'curvature':
        demo_curvature_estimation()
    elif args.mode == 'bbox':
        demo_bounding_box()
    elif args.mode == 'pipeline':
        demo_comprehensive_pipeline()