import os
import shutil
import time

from PIL import Image
import logging
from pathlib import Path
import threading
from paddleocr import PaddleOCR
import requests
import base64
import json
import hashlib
import numpy as np
from wechat_bot import WeChatWorkBot

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageOCRProcessor:
    def __init__(self, source_dir, target_dir, webhook_url, keyword="扫码"):
        """
        初始化图片OCR处理器
        
        Args:
            source_dir (str): 源图片目录
            target_dir (str): 目标图片目录
            webhook_url (str): 企业微信机器人的Webhook地址
            keyword (str): 要搜索的关键词
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.keyword = keyword
        
        # 确保目标目录存在
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保输出目录存在
        output_dir = Path('./output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的图片格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 初始化PaddleOCR，优化配置
        self.ocr = PaddleOCR(
            use_angle_cls=False,  # 关闭方向分类器，因为大多数图片都是正向的
            lang="ch",           # 中文模型
            use_gpu=False,       # 使用CPU
            show_log=False,      # 不显示日志
            use_mp=False,        # 禁用多进程，避免内存分配问题
            total_process_num=1, # 单进程
            det_db_thresh=0.2,   # 进一步降低检测阈值
            det_db_box_thresh=0.2, # 进一步降低检测框阈值
            rec_char_dict_path=None, # 使用默认字典
            cls_model_dir=None,  # 使用默认方向分类器
            det_model_dir=None,  # 使用默认检测模型
            rec_model_dir=None,  # 使用默认识别模型
            cpu_threads=1,       # 单线程
            det_limit_side_len=800,  # 降低检测图片的最大边长
            det_limit_type='min',    # 限制类型为最小边长
            rec_batch_num=1,         # 单张处理
            det_db_unclip_ratio=1.2, # 降低文本框扩张比例
            rec_thresh=0.3,          # 降低识别阈值
            cls_thresh=0.3,          # 降低方向分类阈值
            use_onnx=False,          # 不使用ONNX
            output='./output',       # 指定输出目录
            use_space_char=True,     # 使用空格字符
            drop_score=0.5,          # 降低置信度阈值
        )
        
        # 初始化企业微信机器人
        self.bot = WeChatWorkBot(webhook_url)
    
    def preprocess_image(self, image_path):
        """预处理图片以提高识别速度"""
        try:
            # 打开图片
            with Image.open(image_path) as img:
                # 转换为RGB模式（如果不是的话）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 如果图片太大，进行缩放
                max_size = 800  # 进一步降低最大尺寸以加快处理速度
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 转换为numpy数组
                return np.array(img)
        except Exception as e:
            logger.error(f"预处理图片失败: {str(e)}")
            return None
    
    def process_image(self, image_path):
        """
        处理单个图片，识别文字并检查是否包含关键词
        
        Args:
            image_path (Path): 图片路径
            
        Returns:
            bool: 是否包含关键词
        """
        try:
            # 预处理图片
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return False
            
            # 使用PaddleOCR进行识别
            result = self.ocr.ocr(img_array, cls=True)
            
            # 提取所有识别的文本
            all_text = ""
            if result:
                for line in result[0]:
                    if line[1][0]:  # 确保有识别结果
                        all_text += line[1][0] + "\n"
            
            # 检查是否包含关键词
            if self.keyword in all_text:
                logger.info(f"在图片 {image_path.name} 中找到关键词 '{self.keyword}'")
                return True
            return False
                
        except Exception as e:
            logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
            return False
    
    def move_image(self, image_path):
        """重命名图片并发送到企业微信群"""
        try:
            # 构建新的文件名（添加'd_'前缀）
            new_name = 'd_' + image_path.name
            new_path = image_path.parent / new_name
            
            # 重命名文件
            image_path.rename(new_path)
            logger.info(f"已重命名图片 {image_path.name} 为 {new_name}")
            
            # 发送图片到企业微信群
            self.bot.send_image(str(new_path))
            
        except Exception as e:
            logger.error(f"重命名图片 {image_path} 时出错: {str(e)}")
    
    def process_directory(self):
        """处理源目录中的所有图片"""
        # 获取所有图片文件，只处理不含'd_'前缀的图片
        image_files = [f for f in self.source_dir.iterdir() 
                      if f.is_file() and self.is_image_file(f) 
                      and not f.name.startswith('d_')]
        
        logger.info(f"目录中找到 {len(image_files)} 个未处理的图片文件")
        
        if image_files:
            logger.info(f"新图片列表: {[f.name for f in image_files]}")
            
            # 单张处理图片
            for image_path in image_files:
                start_time = time.time()
                try:
                    # 预处理图片
                    img_array = self.preprocess_image(image_path)
                    if img_array is None:
                        continue
                        
                    # 使用PaddleOCR进行识别
                    result = self.ocr.ocr(img_array, cls=False)
                    
                    # 处理识别结果
                    if result and result[0]:
                        all_text = "".join(line[1][0] + "\n" for line in result[0] if line[1][0])
                        
                        if self.keyword in all_text:
                            logger.info(f"在图片 {image_path.name} 中找到关键词 '{self.keyword}'")
                            self.move_image(image_path)
                        else:
                            self.add_prefix(image_path)
                    else:
                        self.add_prefix(image_path)
                        
                except Exception as e:
                    logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
                    continue
                    
                end_time = time.time()
                print(f"处理图片 {image_path} 耗时: {end_time - start_time} 秒")
        else:
            logger.info("没有找到需要处理的图片")

    def add_prefix(self, image_path):
        """为图片添加d_前缀"""
        try:
            # 构建新的文件名（添加'd_'前缀）
            new_name = 'd_' + image_path.name
            new_path = image_path.parent / new_name
            
            # 重命名文件
            image_path.rename(new_path)
            logger.info(f"已重命名图片 {image_path.name} 为 {new_name}")
            
        except Exception as e:
            logger.error(f"重命名图片 {image_path} 时出错: {str(e)}")

    def is_image_file(self, file_path):
        """检查文件是否为支持的图片格式"""
        return file_path.suffix.lower() in self.supported_formats

def main():
    # 配置源目录和目标目录
    source_dir = "./weibo/img"  # 源图片目录
    target_dir = "./weibo/img"  # 目标图片目录也设置为img目录
    
    # 企业微信机器人的Webhook地址
    webhook_url = os.getenv('WEBHOOK_URL', '')
    if not webhook_url:
        logger.warning("环境变量 WEBHOOK_URL 未设置，将无法发送通知")
    
    # 创建处理器实例
    processor = ImageOCRProcessor(source_dir, target_dir, webhook_url)
    
    while True:
        try:
            # 处理图片
            processor.process_directory()
            # 等待0.5秒
            threading.Event().wait(0.5)
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
            break
        except Exception as e:
            logger.error(f"运行出错: {str(e)}")
            threading.Event().wait(2)  # 发生错误时也等待2秒

if __name__ == "__main__":
    main() 