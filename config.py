# config.py
import os


class Config:
    """配置管理类"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_caption_prompt(cls):
        """获取CAPTION_PROMPT"""
        prompt_path = os.path.join(cls.BASE_DIR, 'prompt_cn_font.txt')
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"提示文件 {prompt_path} 不存在")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()