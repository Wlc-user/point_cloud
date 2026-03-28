FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖（用于OpenCV和科学计算库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建输出目录
RUN mkdir -p output

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认运行示例程序（可以被命令覆盖）
CMD ["python", "-c", "print('工业机器视觉平台已启动！使用 docker run --rm -v $(pwd):/app opencv python examples/complete_pipeline_example.py 运行示例')"]
