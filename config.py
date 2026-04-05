import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    所有的配置项都会自动从同名环境变量或 .env 文件中读取
    """
    # Neo4j 配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j123"
    
    # Ollama 配置
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:4b"
    ollama_embed_model: str = "bge-m3:latest"
    
    # Agent 策略
    max_retries: int = 3
    llm_temperature: float = 0.0
    
    # 读取 .env 文件配置
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# 实例化，全局直接调用 settings 即可
settings = Settings()

# 测试一下是否读取成功
if __name__ == "__main__":
    print(f"成功加载模型: {settings.llm_model}")
    print(f"数据库地址: {settings.neo4j_uri}")