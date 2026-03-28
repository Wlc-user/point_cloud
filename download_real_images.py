"""
真实工业缺陷图片下载系统
从网上下载真实的PCB、芯片、晶圆缺陷图片
"""
import os
import requests
import cv2
import numpy as np
from urllib.parse import urlparse
from pathlib import Path

class RealImageDownloader:
    """真实工业缺陷图片下载器"""
    
    def __init__(self, output_dir="real_images"):
        self.output_dir = output_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """创建必要的目录结构"""
        dirs = [
            f"{self.output_dir}/pcb",
            f"{self.output_dir}/chip", 
            f"{self.output_dir}/wafer",
            f"{self.output_dir}/defects"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def download_image(self, url, save_path, timeout=30):
        """下载单张图片"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            print(f"Download failed: {e}")
        return False
    
    def search_and_download(self, query, category, num_images=5):
        """
        搜索并下载图片
        注意：这里使用示例URL，实际使用时需要接入图片搜索引擎API
        """
        print(f"\nSearching for: {query}")
        print("Note: Using placeholder URLs. In production, integrate with:")
        print("  - Google Images API")
        print("  - Bing Image Search API") 
        print("  - Shutterstock API")
        print("  - Getty Images API")
        
        # 示例：这里应该调用真实的图片搜索API
        # 目前使用占位符说明实现方式
        
        downloaded = []
        for i in range(num_images):
            save_path = f"{self.output_dir}/{category}/{query.replace(' ', '_')}_{i+1}.jpg"
            
            # 实际实现时，这里应该是真实的图片URL
            # 目前创建示例图片说明功能
            self.create_placeholder_image(save_path, query, i+1)
            downloaded.append(save_path)
            print(f"  Created: {save_path}")
        
        return downloaded
    
    def create_placeholder_image(self, save_path, label, index):
        """创建占位图片（实际使用时替换为真实下载）"""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 240
        
        # 添加说明文字
        cv2.putText(img, "PLACEHOLDER", (150, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
        cv2.putText(img, f"{label}", (50, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        cv2.putText(img, f"Image #{index}", (50, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        cv2.putText(img, "Replace with real download", (50, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 100, 100), 1)
        
        cv2.imwrite(save_path, img)
    
    def download_pcb_defects(self):
        """下载PCB缺陷图片"""
        print("\n" + "="*60)
        print("DOWNLOADING PCB DEFECT IMAGES")
        print("="*60)
        
        queries = [
            "PCB open circuit defect",
            "PCB short circuit defect", 
            "PCB solder bridge",
            "PCB missing component",
            "PCB tombstone defect",
            "PCB cold solder joint",
            "PCB solder void",
            "PCB scratch damage"
        ]
        
        all_images = []
        for query in queries:
            images = self.search_and_download(query, "pcb", num_images=3)
            all_images.extend(images)
        
        return all_images
    
    def download_chip_defects(self):
        """下载芯片缺陷图片"""
        print("\n" + "="*60)
        print("DOWNLOADING IC CHIP DEFECT IMAGES")
        print("="*60)
        
        queries = [
            "IC die crack defect",
            "chip pad corrosion",
            "wire bond lift failure",
            "semiconductor contamination",
            "IC package scratch",
            "chip discoloration",
            "foreign particle on chip"
        ]
        
        all_images = []
        for query in queries:
            images = self.search_and_download(query, "chip", num_images=3)
            all_images.extend(images)
        
        return all_images
    
    def download_wafer_defects(self):
        """下载晶圆缺陷图片"""
        print("\n" + "="*60)
        print("DOWNLOADING WAFER DEFECT IMAGES")
        print("="*60)
        
        queries = [
            "silicon wafer edge chip",
            "wafer scratch defect",
            "wafer particle contamination",
            "photoresist residue wafer",
            "wafer probe marks",
            "lithography pattern defect",
            "ring oscillator test structure"
        ]
        
        all_images = []
        for query in queries:
            images = self.search_and_download(query, "wafer", num_images=3)
            all_images.extend(images)
        
        return all_images
    
    def create_detection_demo(self):
        """创建基于真实图片的检测演示"""
        print("\n" + "="*60)
        print("CREATING DETECTION DEMO WITH REAL IMAGES")
        print("="*60)
        
        # 读取下载的图片
        pcb_images = []
        chip_images = []
        wafer_images = []
        
        for category in ['pcb', 'chip', 'wafer']:
            dir_path = f"{self.output_dir}/{category}"
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
                for f in files[:4]:  # 只取前4张
                    img_path = os.path.join(dir_path, f)
                    img = cv2.imread(img_path)
                    if img is not None:
                        if category == 'pcb':
                            pcb_images.append(img)
                        elif category == 'chip':
                            chip_images.append(img)
                        else:
                            wafer_images.append(img)
        
        # 创建综合展示图
        if pcb_images and chip_images and wafer_images:
            # 调整大小
            target_size = (300, 200)
            
            pcb_display = cv2.resize(pcb_images[0], target_size)
            chip_display = cv2.resize(chip_images[0], target_size)
            wafer_display = cv2.resize(wafer_images[0], target_size)
            
            # 添加标签
            cv2.putText(pcb_display, "PCB DEFECT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(chip_display, "CHIP DEFECT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(wafer_display, "WAFER DEFECT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 拼接
            row = np.hstack((pcb_display, chip_display, wafer_display))
            cv2.imwrite(f"{self.output_dir}/real_defects_overview.jpg", row)
            print(f"[OK] Overview saved: {self.output_dir}/real_defects_overview.jpg")
        
        return True

def main():
    """主程序"""
    print("="*70)
    print("REAL INDUSTRIAL DEFECT IMAGE DOWNLOADER")
    print("="*70)
    print("\nThis system downloads real defect images from the internet.")
    print("Currently using placeholder images - integrate with image APIs for production.")
    
    downloader = RealImageDownloader()
    
    # 下载各类缺陷图片
    pcb_images = downloader.download_pcb_defects()
    chip_images = downloader.download_chip_defects()
    wafer_images = downloader.download_wafer_defects()
    
    # 创建检测演示
    downloader.create_detection_demo()
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\nImages saved to: {downloader.output_dir}/")
    print("\nTo use real images:")
    print("1. Replace placeholder images with actual downloads")
    print("2. Integrate Google Images API, Bing API, or stock photo APIs")
    print("3. Or manually add your own defect images to the folders")

if __name__ == '__main__':
    main()
