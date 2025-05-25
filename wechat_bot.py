import base64
import hashlib
import logging
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeChatWorkBot:
    def __init__(self, webhook_url):
        """
        初始化企业微信机器人
        
        Args:
            webhook_url (str): 企业微信机器人的Webhook地址
        """
        self.webhook_url = webhook_url
    
    def send_image(self, image_path, title="发现扫码图片"):
        """
        发送图片到企业微信群
        
        Args:
            image_path (str): 图片路径
            title (str): 图片标题
        """
        try:
            # 读取图片文件
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 将图片转换为base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # 构建消息数据
            data = {
                "msgtype": "image",
                "image": {
                    "base64": base64_image,
                    "md5": self._calculate_md5(image_data)
                }
            }
            
            # 发送请求
            response = requests.post(self.webhook_url, json=data)
            response.raise_for_status()
            
            logger.info(f"成功发送图片到企业微信群: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"发送图片到企业微信群失败: {str(e)}")
            return False
    
    def _calculate_md5(self, data):
        """计算数据的MD5值"""
        return hashlib.md5(data).hexdigest() 