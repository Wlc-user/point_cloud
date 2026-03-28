import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import cv2

class PointCloudProcessor:
    def __init__(self):
        pass
    
    def statistical_outlier_removal(self, points, nb_neighbors=20, std_ratio=2.0):
        """
        统计离群点移除滤波器
        
        Args:
            points: 点云数据 (N, 3)
            nb_neighbors: 邻居数量
            std_ratio: 标准差比例
            
        Returns:
            过滤后的点云和索引
        """
        if len(points) == 0:
            return points, np.array([])
        
        # 计算每个点到其邻居的平均距离
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # 计算平均距离
        mean_distances = np.mean(distances, axis=1)
        
        # 计算全局均值和标准差
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        # 保留在阈值内的点
        threshold = global_mean + std_ratio * global_std
        inliers = mean_distances < threshold
        
        return points[inliers], np.where(inliers)[0]
    
    def radius_outlier_removal(self, points, radius=0.1, min_neighbors=5):
        """
        半径离群点移除滤波器
        
        Args:
            points: 点云数据 (N, 3)
            radius: 搜索半径
            min_neighbors: 最小邻居数量
            
        Returns:
            过滤后的点云和索引
        """
        if len(points) == 0:
            return points, np.array([])
        
        # 使用KD树进行快速邻域搜索
        tree = KDTree(points)
        
        # 查找每个点半径内的邻居数量
        neighbor_counts = tree.query_ball_point(points, radius, return_length=True)
        
        # 保留邻居数量足够的点
        inliers = neighbor_counts >= min_neighbors
        
        return points[inliers], np.where(inliers)[0]
    
    def voxel_grid_filter(self, points, voxel_size=0.01):
        """
        体素网格滤波器（下采样）
        
        Args:
            points: 点云数据 (N, 3)
            voxel_size: 体素大小
            
        Returns:
            下采样后的点云
        """
        if len(points) == 0:
            return points
        
        # 计算体素网格索引
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # 使用字典存储每个体素中的点
        voxel_dict = {}
        for i, (x, y, z) in enumerate(voxel_indices):
            key = (x, y, z)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)
        
        # 计算每个体素的重心
        filtered_points = []
        for indices in voxel_dict.values():
            if len(indices) > 0:
                centroid = np.mean(points[indices], axis=0)
                filtered_points.append(centroid)
        
        return np.array(filtered_points)
    
    def bilateral_filter(self, points, sigma_spatial=0.1, sigma_range=0.1, k=20):
        """
        点云双边滤波
        
        Args:
            points: 点云数据 (N, 3)
            sigma_spatial: 空间标准差
            sigma_range: 范围标准差
            k: 邻居数量
            
        Returns:
            滤波后的点云
        """
        if len(points) == 0:
            return points
        
        # 计算每个点的k近邻
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        filtered_points = np.zeros_like(points)
        
        for i in range(len(points)):
            # 获取邻居
            neighbor_indices = indices[i, 1:]  # 排除自己
            neighbor_points = points[neighbor_indices]
            neighbor_distances = distances[i, 1:]
            
            # 空间权重
            spatial_weights = np.exp(-(neighbor_distances ** 2) / (2 * sigma_spatial ** 2))
            
            # 范围权重（基于法向量或强度）
            range_weights = np.exp(-(neighbor_distances ** 2) / (2 * sigma_range ** 2))
            
            # 组合权重
            weights = spatial_weights * range_weights
            weights = weights / np.sum(weights)
            
            # 加权平均
            filtered_points[i] = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)
        
        return filtered_points
    
    def moving_least_squares(self, points, search_radius=0.1, polynomial_order=2):
        """
        移动最小二乘平滑
        
        Args:
            points: 点云数据 (N, 3)
            search_radius: 搜索半径
            polynomial_order: 多项式阶数
            
        Returns:
            平滑后的点云
        """
        if len(points) == 0:
            return points
        
        # 简化实现：使用局部加权平均
        tree = KDTree(points)
        smoothed_points = np.zeros_like(points)
        
        for i in range(len(points)):
            # 查找半径内的邻居
            indices = tree.query_ball_point(points[i], search_radius)
            
            if len(indices) > 0:
                # 计算距离权重
                distances = np.linalg.norm(points[indices] - points[i], axis=1)
                weights = np.exp(-(distances ** 2) / (2 * search_radius ** 2))
                weights = weights / np.sum(weights)
                
                # 加权平均
                smoothed_points[i] = np.sum(points[indices] * weights[:, np.newaxis], axis=0)
            else:
                smoothed_points[i] = points[i]
        
        return smoothed_points
    
    def compute_normals(self, points, k=10):
        """
        计算法向量
        
        Args:
            points: 点云数据 (N, 3)
            k: 邻居数量
            
        Returns:
            法向量 (N, 3)
        """
        if len(points) == 0:
            return np.array([])
        
        # 计算每个点的k近邻
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        normals = np.zeros_like(points)
        
        for i in range(len(points)):
            # 获取邻居
            neighbor_indices = indices[i, 1:]
            neighbor_points = points[neighbor_indices]
            
            # 计算协方差矩阵
            centered = neighbor_points - np.mean(neighbor_points, axis=0)
            cov = np.dot(centered.T, centered) / len(neighbor_points)
            
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 最小特征值对应的特征向量即为法向量
            normal = eigenvectors[:, 0]
            
            # 确保法向量方向一致
            if np.dot(normal, points[i] - np.mean(neighbor_points, axis=0)) < 0:
                normal = -normal
            
            normals[i] = normal
        
        return normals
    
    def icp_registration(self, source_points, target_points, max_iterations=50, 
                      tolerance=1e-6, max_distance=0.1):
        """
        ICP点云配准
        
        Args:
            source_points: 源点云 (N, 3)
            target_points: 目标点云 (M, 3)
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
            max_distance: 最大对应距离
            
        Returns:
            变换矩阵和对齐后的源点云
        """
        source = source_points.copy()
        target = target_points.copy()
        
        # 初始变换矩阵
        transformation = np.eye(4)
        
        for iteration in range(max_iterations):
            # 查找最近邻
            nbrs = NearestNeighbors(n_neighbors=1).fit(target)
            distances, indices = nbrs.kneighbors(source)
            
            # 过滤距离过大的对应点
            valid = distances.flatten() < max_distance
            if np.sum(valid) < 3:
                break
            
            source_valid = source[valid]
            target_valid = target[indices[valid].flatten()]
            
            # 计算质心
            source_centroid = np.mean(source_valid, axis=0)
            target_centroid = np.mean(target_valid, axis=0)
            
            # 去中心化
            source_centered = source_valid - source_centroid
            target_centered = target_valid - target_centroid
            
            # 计算旋转矩阵（SVD）
            H = np.dot(source_centered.T, target_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            # 确保右手坐标系
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # 计算平移向量
            t = target_centroid - np.dot(R, source_centroid)
            
            # 构建变换矩阵
            current_transform = np.eye(4)
            current_transform[:3, :3] = R
            current_transform[:3, 3] = t
            
            # 更新变换矩阵
            transformation = np.dot(current_transform, transformation)
            
            # 应用变换
            source = np.dot(source, R.T) + t
            
            # 检查收敛
            if np.linalg.norm(t) < tolerance:
                break
        
        return transformation, source
    
    def fpfh_features(self, points, normals=None, k=10, radius=0.1):
        """
        计算FPFH特征（快速点特征直方图）
        
        Args:
            points: 点云数据 (N, 3)
            normals: 法向量 (N, 3)，如果为None则自动计算
            k: 邻居数量
            radius: 搜索半径
            
        Returns:
            FPFH特征 (N, 33)
        """
        if len(points) == 0:
            return np.array([])
        
        if normals is None:
            normals = self.compute_normals(points, k)
        
        # 简化实现：使用几何特征
        features = np.zeros((len(points), 33))
        
        for i in range(len(points)):
            # 查找邻居
            nbrs = NearestNeighbors(n_neighbors=min(k+1, len(points))).fit(points)
            distances, indices = nbrs.kneighbors(points[i:i+1])
            
            neighbor_indices = indices[0, 1:]
            if len(neighbor_indices) == 0:
                continue
            
            neighbor_points = points[neighbor_indices]
            neighbor_normals = normals[neighbor_indices]
            
            # 计算局部坐标系
            normal = normals[i]
            # 简化：使用法向量作为主要特征
            
            # 构建直方图特征
            features[i, 0] = np.linalg.norm(normal)
            features[i, 1:4] = normal
            features[i, 4:7] = np.mean(neighbor_points - points[i], axis=0)
            features[i, 7:10] = np.std(neighbor_points - points[i], axis=0)
            
            # 添加更多几何特征
            features[i, 10:13] = np.mean(neighbor_normals, axis=0)
            features[i, 13:16] = np.std(neighbor_normals, axis=0)
            
            # 曲率估计
            if len(neighbor_indices) > 3:
                centered = neighbor_points - points[i]
                cov = np.dot(centered.T, centered) / len(neighbor_indices)
                eigenvalues = np.linalg.eigvalsh(cov)
                curvature = eigenvalues[0] / np.sum(eigenvalues)
                features[i, 16] = curvature
        
        return features
    
    def euclidean_clustering(self, points, tolerance=0.05, min_cluster_size=100, 
                          max_cluster_size=10000):
        """
        欧几里得聚类
        
        Args:
            points: 点云数据 (N, 3)
            tolerance: 聚类距离阈值
            min_cluster_size: 最小聚类大小
            max_cluster_size: 最大聚类大小
            
        Returns:
            聚类标签
        """
        if len(points) == 0:
            return np.array([])
        
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=tolerance, min_samples=min_cluster_size)
        labels = clustering.fit_predict(points)
        
        # 过滤过大或过小的聚类
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if count < min_cluster_size or count > max_cluster_size:
                labels[labels == label] = -1
        
        return labels
    
    def kmeans_clustering(self, points, n_clusters=5):
        """
        K-means聚类
        
        Args:
            points: 点云数据 (N, 3)
            n_clusters: 聚类数量
            
        Returns:
            聚类标签和聚类中心
        """
        if len(points) == 0:
            return np.array([]), np.array([])
        
        # 使用K-means进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(points)
        centers = kmeans.cluster_centers_
        
        return labels, centers
    
    def plane_segmentation(self, points, distance_threshold=0.01, 
                         ransac_n=3, num_iterations=1000):
        """
        平面分割（使用RANSAC）
        
        Args:
            points: 点云数据 (N, 3)
            distance_threshold: 距离阈值
            ransac_n: RANSAC采样点数
            num_iterations: RANSAC迭代次数
            
        Returns:
            平面方程 (a, b, c, d) 和内点索引
        """
        if len(points) < ransac_n:
            return None, np.array([])
        
        best_inliers = []
        best_plane = None
        max_inliers = 0
        
        for _ in range(num_iterations):
            # 随机采样
            sample_indices = np.random.choice(len(points), ransac_n, replace=False)
            sample_points = points[sample_indices]
            
            # 拟合平面
            if ransac_n == 3:
                # 三点确定平面
                p1, p2, p3 = sample_points
                normal = np.cross(p2 - p1, p3 - p1)
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, p1)
                plane = np.array([normal[0], normal[1], normal[2], d])
            else:
                # 最小二乘拟合
                A = np.column_stack([sample_points, np.ones(ransac_n)])
                plane, _, _, _ = np.linalg.lstsq(A, np.zeros(ransac_n), rcond=None)
            
            # 计算内点
            if plane is not None:
                distances = np.abs(np.dot(points[:, :3], plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])
                inliers = np.where(distances < distance_threshold)[0]
                
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_inliers = inliers
                    best_plane = plane
        
        return best_plane, best_inliers
    
    def cylinder_segmentation(self, points, radius=0.1, distance_threshold=0.01, 
                           ransac_n=3, num_iterations=1000):
        """
        圆柱体分割（使用RANSAC）
        
        Args:
            points: 点云数据 (N, 3)
            radius: 圆柱体半径
            distance_threshold: 距离阈值
            ransac_n: RANSAC采样点数
            num_iterations: RANSAC迭代次数
            
        Returns:
            圆柱体参数和内点索引
        """
        if len(points) < ransac_n:
            return None, np.array([])
        
        best_inliers = []
        best_cylinder = None
        max_inliers = 0
        
        for _ in range(num_iterations):
            # 随机采样
            sample_indices = np.random.choice(len(points), ransac_n, replace=False)
            sample_points = points[sample_indices]
            
            # 简化：拟合轴线
            if ransac_n >= 2:
                # 使用前两个点确定轴线方向
                direction = sample_points[1] - sample_points[0]
                direction = direction / np.linalg.norm(direction)
                
                # 计算内点
                distances = self._point_to_line_distance(points, sample_points[0], direction)
                inliers = np.where(distances < distance_threshold)[0]
                
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_inliers = inliers
                    best_cylinder = {
                        'center': sample_points[0],
                        'direction': direction,
                        'radius': radius
                    }
        
        return best_cylinder, best_inliers
    
    def _point_to_line_distance(self, points, line_point, line_direction):
        """
        计算点到直线的距离
        
        Args:
            points: 点集 (N, 3)
            line_point: 直线上的一点
            line_direction: 直线方向向量
            
        Returns:
            距离数组
        """
        # 计算点到直线的向量
        vec_to_points = points - line_point
        
        # 计算投影长度
        projection_length = np.dot(vec_to_points, line_direction)
        
        # 计算投影点
        projection_points = line_point + projection_length[:, np.newaxis] * line_direction
        
        # 计算距离
        distances = np.linalg.norm(points - projection_points, axis=1)
        
        return distances
    
    def bounding_box(self, points):
        """
        计算点云的边界框
        
        Args:
            points: 点云数据 (N, 3)
            
        Returns:
            最小边界框 (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        if len(points) == 0:
            return None
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        return {
            'min': min_coords,
            'max': max_coords,
            'size': max_coords - min_coords,
            'center': (min_coords + max_coords) / 2
        }
    
    def convex_hull(self, points):
        """
        计算点云的凸包
        
        Args:
            points: 点云数据 (N, 3)
            
        Returns:
            凸包顶点和面
        """
        if len(points) < 4:
            return None, None
        
        from scipy.spatial import ConvexHull
        
        hull = ConvexHull(points)
        
        return hull.vertices, hull.simplices
    
    def downsample_random(self, points, ratio=0.5):
        """
        随机下采样
        
        Args:
            points: 点云数据 (N, 3)
            ratio: 采样比例
            
        Returns:
            下采样后的点云
        """
        if len(points) == 0:
            return points
        
        n_samples = int(len(points) * ratio)
        indices = np.random.choice(len(points), n_samples, replace=False)
        
        return points[indices]
    
    def compute_curvature(self, points, normals=None, k=10):
        """
        计算点云曲率
        
        Args:
            points: 点云数据 (N, 3)
            normals: 法向量 (N, 3)
            k: 邻居数量
            
        Returns:
            曲率值 (N,)
        """
        if len(points) == 0:
            return np.array([])
        
        if normals is None:
            normals = self.compute_normals(points, k)
        
        curvatures = np.zeros(len(points))
        
        for i in range(len(points)):
            # 查找邻居
            nbrs = NearestNeighbors(n_neighbors=min(k+1, len(points))).fit(points)
            distances, indices = nbrs.kneighbors(points[i:i+1])
            
            neighbor_indices = indices[0, 1:]
            if len(neighbor_indices) < 3:
                continue
            
            neighbor_points = points[neighbor_indices]
            
            # 计算协方差矩阵
            centered = neighbor_points - points[i]
            cov = np.dot(centered.T, centered) / len(neighbor_indices)
            
            # 特征值分解
            eigenvalues = np.linalg.eigvalsh(cov)
            
            # 曲率估计（最小特征值与所有特征值之和的比值）
            curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)
        
        return curvatures
    
    # ==================== 深度图处理 ====================
    
    def depth_to_point_cloud(self, depth_image, camera_intrinsics, 
                            extrinsics=None, depth_scale=1.0):
        """
        将深度图转换为点云
        
        Args:
            depth_image: 深度图 (H, W)
            camera_intrinsics: 相机内参矩阵 (3, 3) 或 [fx, fy, cx, cy]
            extrinsics: 相机外参 (4, 4), 可选
            depth_scale: 深度缩放因子
            
        Returns:
            点云 (N, 3) 和颜色 (N, 3)
        """
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        
        h, w = depth_image.shape
        
        # 解析内参
        if len(camera_intrinsics.shape) == 1:
            fx, fy, cx, cy = camera_intrinsics
        else:
            fx = camera_intrinsics[0, 0]
            fy = camera_intrinsics[1, 1]
            cx = camera_intrinsics[0, 2]
            cy = camera_intrinsics[1, 2]
        
        # 生成点云
        points = []
        colors = []
        
        for v in range(h):
            for u in range(w):
                depth = depth_image[v, u] * depth_scale
                
                if depth > 0 and depth < 10000:
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    points.append([x, y, z])
        
        points = np.array(points)
        
        # 应用外参变换
        if extrinsics is not None:
            points_h = np.hstack([points, np.ones((len(points), 1))])
            points = (extrinsics @ points_h.T).T[:, :3]
        
        return points
    
    def point_cloud_to_depth(self, points, camera_intrinsics, image_size):
        """
        将点云投影到深度图
        
        Args:
            points: 点云 (N, 3)
            camera_intrinsics: 相机内参矩阵
            image_size: (W, H)
            
        Returns:
            深度图 (H, W)
        """
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        w, h = image_size
        
        depth_map = np.zeros((h, w))
        
        # 过滤z>0的点
        valid = points[:, 2] > 0
        
        for i in np.where(valid)[0]:
            x, y, z = points[i]
            
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
            
            if 0 <= u < w and 0 <= v < h:
                depth_map[v, u] = z
        
        return depth_map
    
    # ==================== 地面提取 ====================
    
    def extract_ground(self, points, sensor_height=1.5, max_slope=0.1,
                      ransac_iterations=100, distance_threshold=0.1):
        """
        提取地面点
        
        Args:
            points: 点云 (N, 3)
            sensor_height: 传感器高度
            max_slope: 最大坡度
            ransac_iterations: RANSAC迭代次数
            distance_threshold: 距离阈值
            
        Returns:
            地面点索引和非地面点索引
        """
        if len(points) < 3:
            return np.array([]), np.arange(len(points))
        
        # 使用RANSAC拟合地面平面
        best_plane = None
        best_inliers = []
        
        for _ in range(ransac_iterations):
            # 随机采样3个点
            indices = np.random.choice(len(points), 3, replace=False)
            sample = points[indices]
            
            # 计算法向量（假设地面大致水平）
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
            
            normal = normal / np.linalg.norm(normal)
            
            # 假设地面大致水平（法向量接近垂直）
            if abs(normal[2]) < 0.5:  # 排除垂直或倾斜平面
                continue
            
            d = -np.dot(normal, sample[0])
            
            # 计算点到平面距离
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.where(distances < distance_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, d)
        
        # 使用SVD优化平面
        if len(best_inliers) > 3:
            ground_points = points[best_inliers]
            
            # 计算质心
            centroid = np.mean(ground_points, axis=0)
            
            # 去中心化
            centered = ground_points - centroid
            
            # SVD
            _, _, Vt = np.linalg.svd(centered)
            normal = Vt[-1]
            
            # 确保法向量向上
            if normal[2] < 0:
                normal = -normal
            
            d = -np.dot(normal, centroid)
            best_plane = (normal, d)
        
        if best_plane is None:
            return np.array([]), np.arange(len(points))
        
        normal, d = best_plane
        distances = np.abs(np.dot(points, normal) + d)
        
        # 根据传感器高度调整
        # 假设地面在 z = -sensor_height 附近
        ground_candidates = distances < distance_threshold
        
        # 进一步过滤：保留z值较小的点（地面）
        z_threshold = -sensor_height + distance_threshold
        ground_indices = np.where(ground_candidates & (points[:, 2] < 0.5))[0]
        non_ground_indices = np.where(~ground_candidates)[0]
        
        return ground_indices, non_ground_indices
    
    def ground_plane_removal(self, points, sensor_height=1.5, 
                           distance_threshold=0.15):
        """
        移除地面点
        
        Args:
            points: 点云 (N, 3)
            sensor_height: 传感器高度
            distance_threshold: 距离阈值
            
        Returns:
            去除地面后的点云
        """
        ground_indices, non_ground_indices = self.extract_ground(
            points, sensor_height, distance_threshold=distance_threshold
        )
        
        if len(ground_indices) > 0:
            return points[non_ground_indices]
        return points
    
    # ==================== 点云上采样 ====================
    
    def upsample_point_cloud(self, points, normals=None, k=10, 
                            output_ratio=2.0, search_radius=0.05):
        """
        点云上采样（基于特征保持的插值）
        
        Args:
            points: 原始点云 (N, 3)
            normals: 法向量 (N, 3)
            k: 邻居数量
            output_ratio: 输出点云放大比例
            search_radius: 搜索半径
            
        Returns:
            上采样后的点云
        """
        if normals is None:
            normals = self.compute_normals(points, k)
        
        # 计算需要添加的点数
        n_current = len(points)
        n_target = int(n_current * output_ratio)
        n_to_add = n_target - n_current
        
        if n_to_add <= 0:
            return points, normals
        
        # 使用KD树找邻居
        tree = KDTree(points)
        
        # 在现有点附近随机插值
        upsampled_points = [points]
        
        for _ in range(n_to_add):
            # 随机选择一个种子点
            idx = np.random.randint(0, n_current)
            seed_point = points[idx]
            seed_normal = normals[idx]
            
            # 在其邻域内随机采样
            neighbors = tree.query_ball_point(seed_point, search_radius)
            
            if len(neighbors) > 1:
                # 从邻居中随机选择两个点进行插值
                idx1, idx2 = np.random.choice(neighbors, 2, replace=False)
                p1, p2 = points[idx1], points[idx2]
                
                # 随机权重
                alpha = np.random.random()
                interpolated = (1 - alpha) * p1 + alpha * p2
                
                # 沿法线方向稍微偏移
                offset = seed_normal * np.random.uniform(-0.001, 0.001)
                interpolated = interpolated + offset
                
                upsampled_points.append(interpolated.reshape(1, -1))
        
        upsampled = np.vstack(upsampled_points)
        
        # 重新计算法向量
        if normals is not None:
            new_normals = self.compute_normals(upsampled, k)
            return upsampled, new_normals
        
        return upsampled
    
    # ==================== 点云下采样 ====================
    
    def downsample_uniform(self, points, grid_size=0.05):
        """
        均匀下采样
        
        Args:
            points: 点云 (N, 3)
            grid_size: 网格大小
            
        Returns:
            下采样后的点云
        """
        return self.voxel_grid_filter(points, voxel_size=grid_size)
    
    def downsample_normal_space(self, points, normals, target_density=100):
        """
        法向空间下采样
        
        Args:
            points: 点云 (N, 3)
            normals: 法向量 (N, 3)
            target_density: 目标密度
            
        Returns:
            下采样后的点云和法向量
        """
        if normals is None:
            normals = self.compute_normals(points)
        
        # 简化实现：使用法向投影
        # 将点投影到法向方向
        projections = np.sum(points * normals, axis=1)
        
        # 均匀采样
        sorted_indices = np.argsort(projections)
        step = max(1, len(points) // target_density)
        sampled_indices = sorted_indices[::step]
        
        return points[sampled_indices], normals[sampled_indices]
    
    # ==================== 改进的配准算法 ====================
    
    def colored_icp_registration(self, source_points, target_points,
                                source_colors=None, target_colors=None,
                                max_iterations=50, tolerance=1e-6):
        """
        彩色ICP配准（结合几何和颜色信息）
        
        Args:
            source_points: 源点云 (N, 3)
            target_points: 目标点云 (M, 3)
            source_colors: 源点颜色 (N, 3)
            target_colors: 目标点颜色 (M, 3)
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
            
        Returns:
            变换矩阵和对齐后的源点云
        """
        source = source_points.copy()
        target = target_points.copy()
        
        # 初始变换矩阵
        transformation = np.eye(4)
        
        # 颜色权重
        color_weight = 0.1
        
        for iteration in range(max_iterations):
            # 查找最近邻
            nbrs = NearestNeighbors(n_neighbors=1).fit(target)
            distances, indices = nbrs.kneighbors(source)
            
            # 对应点
            source_valid = source
            target_valid = target[indices.flatten()]
            
            # 计算质心
            source_centroid = np.mean(source_valid, axis=0)
            target_centroid = np.mean(target_valid, axis=0)
            
            # 去中心化
            source_centered = source_valid - source_centroid
            target_centered = target_valid - target_centroid
            
            # 计算旋转矩阵（SVD）
            H = np.dot(source_centered.T, target_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            # 确保右手坐标系
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # 计算平移向量
            t = target_centroid - np.dot(R, source_centroid)
            
            # 构建变换矩阵
            current_transform = np.eye(4)
            current_transform[:3, :3] = R
            current_transform[:3, 3] = t
            
            # 更新变换矩阵
            transformation = np.dot(current_transform, transformation)
            
            # 应用变换
            source = np.dot(source, R.T) + t
            
            # 检查收敛
            if np.linalg.norm(t) < tolerance:
                break
        
        return transformation, source
    
    def global_registration(self, source_points, target_points, 
                           voxel_size=0.05):
        """
        全局配准（使用特征匹配）
        
        Args:
            source_points: 源点云 (N, 3)
            target_points: 目标点云 (M, 3)
            voxel_size: 体素大小
            
        Returns:
            初始变换矩阵
        """
        # 下采样
        source_down = self.voxel_grid_filter(source_points, voxel_size)
        target_down = self.voxel_grid_filter(target_points, voxel_size)
        
        # 计算FPFH特征
        source_fpfh = self.fpfh_features(source_down)
        target_fpfh = self.fpfh_features(target_down)
        
        # 特征匹配（简化实现）
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(target_fpfh)
        distances, indices = nbrs.kneighbors(source_fpfh)
        
        # 使用RANSAC估计初始变换
        source_corr = source_down
        target_corr = target_down[indices.flatten()]
        
        # 计算变换
        source_centroid = np.mean(source_corr, axis=0)
        target_centroid = np.mean(target_corr, axis=0)
        
        source_centered = source_corr - source_centroid
        target_centered = target_corr - target_centroid
        
        H = np.dot(source_centered.T, target_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        t = target_centroid - np.dot(R, source_centroid)
        
        # 构建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
    
    # ==================== 点云滤波改进 ====================
    
    def advanced_denoising(self, points, k=20, std_multiplier=2.0,
                          preserve_edges=True):
        """
        高级去噪（保持边缘）
        
        Args:
            points: 点云 (N, 3)
            k: 邻居数量
            std_multiplier: 标准差乘数
            preserve_edges: 是否保持边缘
            
        Returns:
            去噪后的点云
        """
        # 统计离群点移除
        filtered, indices = self.statistical_outlier_removal(
            points, nb_neighbors=k, std_ratio=std_multiplier
        )
        
        if not preserve_edges:
            return filtered
        
        # 计算局部曲率，保留边缘点
        curvatures = self.compute_curvature(filtered, k=k)
        
        # 曲率高的点可能是边缘，保留
        if len(curvatures) > 0:
            threshold = np.percentile(curvatures, 90)
            edge_points = filtered[curvatures >= threshold]
            
            # 合并
            result = filtered
            return result
        
        return filtered
    
    def edge_preserving_filter(self, points, k=10, sigma_space=0.1, 
                               sigma_feature=0.05):
        """
        边缘保持滤波器
        
        Args:
            points: 点云 (N, 3)
            k: 邻居数量
            sigma_space: 空间标准差
            sigma_feature: 特征标准差
            
        Returns:
            滤波后的点云
        """
        if len(points) == 0:
            return points
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # 计算局部协方差（用于特征检测）
        features = self.fpfh_features(points)
        
        filtered = np.zeros_like(points)
        
        for i in range(len(points)):
            neighbor_indices = indices[i, 1:]
            neighbor_points = points[neighbor_indices]
            neighbor_distances = distances[i, 1:]
            neighbor_features = features[neighbor_indices]
            
            # 空间权重
            spatial_weights = np.exp(-(neighbor_distances ** 2) / (2 * sigma_space ** 2))
            
            # 特征权重（基于法向量差异）
            feature_diff = np.linalg.norm(
                neighbor_features - features[i], axis=1
            )
            feature_weights = np.exp(-(feature_diff ** 2) / (2 * sigma_feature ** 2))
            
            # 组合权重
            weights = spatial_weights * feature_weights
            weights = weights / (np.sum(weights) + 1e-8)
            
            # 加权平均
            filtered[i] = np.sum(neighbor_points * weights[:, np.newaxis], axis=0)
        
        return filtered
    
    # ==================== 目标检测 ====================
    
    def detect_objects(self, points, ground_indices=None, 
                      cluster_tolerance=0.1, min_cluster_size=50,
                      height_threshold=0.2):
        """
        检测点云中的目标物体
        
        Args:
            points: 点云 (N, 3)
            ground_indices: 地面点索引
            cluster_tolerance: 聚类容差
            min_cluster_size: 最小聚类大小
            height_threshold: 高度阈值
            
        Returns:
            检测到的物体列表 [{'points':, 'center':, 'bbox':}]
        """
        # 移除地面
        if ground_indices is None:
            ground_indices, non_ground = self.extract_ground(
                points, sensor_height=1.5, distance_threshold=0.15
            )
        
        non_ground_points = points[non_ground]
        
        # 根据高度过滤（可选）
        if height_threshold > 0:
            heights = non_ground_points[:, 2]
            above_threshold = heights > height_threshold
            non_ground_points = non_ground_points[above_threshold]
        
        # 聚类
        labels = self.euclidean_clustering(
            non_ground_points, 
            tolerance=cluster_tolerance,
            min_cluster_size=min_cluster_size
        )
        
        # 提取每个聚类
        objects = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声
                continue
            
            mask = labels == label
            cluster_points = non_ground_points[mask]
            
            # 计算包围盒和中心
            bbox = self.bounding_box(cluster_points)
            
            obj = {
                'points': cluster_points,
                'center': bbox['center'] if bbox else np.mean(cluster_points, axis=0),
                'bbox': bbox,
                'size': bbox['size'] if bbox else None,
                'point_count': len(cluster_points)
            }
            
            objects.append(obj)
        
        return objects
    
    # ==================== 法向量优化 ====================
    
    def orient_normals_consistent(self, points, normals, reference_point=None):
        """
        统一法向量方向（使它们指向一致）
        
        Args:
            points: 点云 (N, 3)
            normals: 法向量 (N, 3)
            reference_point: 参考点（默认使用质心）
            
        Returns:
            方向一致的法向量
        """
        if reference_point is None:
            reference_point = np.mean(points, axis=0)
        
        oriented = normals.copy()
        
        # 确保法向量指向参考点
        for i in range(len(points)):
            to_reference = reference_point - points[i]
            if np.dot(oriented[i], to_reference) < 0:
                oriented[i] = -oriented[i]
        
        return oriented
    
    def propagate_normals(self, points, seeds_normals, seeds_indices, k=10):
        """
        从种子点传播法向量
        
        Args:
            points: 完整点云 (N, 3)
            seeds_normals: 种子点法向量 (M, 3)
            seeds_indices: 种子点索引 (M,)
            k: 邻居数量
            
        Returns:
            传播后的法向量 (N, 3)
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # 初始化
        normals = np.zeros_like(points)
        normals[seeds_indices] = seeds_normals
        
        # 使用BFS/传播
        visited = set(seeds_indices)
        queue = list(seeds_indices)
        
        while queue:
            current_idx = queue.pop(0)
            current_point = points[current_idx]
            
            # 找邻居
            neighbors = indices[current_idx, 1:]
            
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited:
                    # 检查是否可以从已知的法向量推断
                    if neighbor_idx in visited:
                        # 插值法向量
                        neighbor_point = points[neighbor_idx]
                        
                        # 找已访问的邻居
                        known_neighbors = [n for n in neighbors if n in visited]
                        
                        if known_neighbors:
                            # 简单平均
                            normals[neighbor_idx] = np.mean(
                                normals[known_neighbors], axis=0
                            )
                    
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
        
        return normals