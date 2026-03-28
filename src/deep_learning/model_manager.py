import os
import json
import time

class ModelManager:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
    def register_model(self, model_path, model_name, model_type, description=''):
        """
        注册模型
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
            model_type: 模型类型
            description: 模型描述
            
        Returns:
            模型ID
        """
        model_id = f"{model_name}_{int(time.time())}"
        model_info = {
            'id': model_id,
            'name': model_name,
            'type': model_type,
            'description': description,
            'path': model_path,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '1.0.0'
        }
        
        # 保存模型信息
        info_path = os.path.join(self.models_dir, f"{model_id}.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        return model_id
    
    def get_model_info(self, model_id):
        """
        获取模型信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型信息
        """
        info_path = os.path.join(self.models_dir, f"{model_id}.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        return None
    
    def list_models(self):
        """
        列出所有模型
        
        Returns:
            模型列表
        """
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.models_dir, file), 'r') as f:
                    models.append(json.load(f))
        return models
    
    def update_model(self, model_id, **kwargs):
        """
        更新模型信息
        
        Args:
            model_id: 模型ID
            **kwargs: 更新的参数
        """
        info_path = os.path.join(self.models_dir, f"{model_id}.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            
            model_info.update(kwargs)
            model_info['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    def delete_model(self, model_id):
        """
        删除模型
        
        Args:
            model_id: 模型ID
        """
        info_path = os.path.join(self.models_dir, f"{model_id}.json")
        if os.path.exists(info_path):
            os.remove(info_path)
        
        # 删除模型文件
        model_info = self.get_model_info(model_id)
        if model_info and os.path.exists(model_info.get('path', '')):
            os.remove(model_info['path'])
    
    def get_model_by_type(self, model_type):
        """
        根据类型获取模型
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型列表
        """
        models = self.list_models()
        return [model for model in models if model.get('type') == model_type]
    
    def get_latest_model(self, model_name):
        """
        获取最新版本的模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            最新模型信息
        """
        models = self.list_models()
        model_versions = [model for model in models if model.get('name') == model_name]
        if model_versions:
            return sorted(model_versions, key=lambda x: x.get('created_at'), reverse=True)[0]
        return None