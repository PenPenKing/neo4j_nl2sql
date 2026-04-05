import sys
from pathlib import Path

# 项目根目录加入 sys.path，否则在 core/ 下运行或调试时无法找到根目录的 config.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, DriverError
import traceback
from config import settings

class Neo4jExecutor:
    def __init__(self):
        """
        从 config.py 中读取 settings 自动初始化驱动
        """
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, 
            auth=(settings.neo4j_user, settings.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def run_query(self, cypher_code: str):
        """
        核心方法：执行 Cypher 并返回结果或错误信息
        """
        # 预处理：去掉 LLM 可能多吐出来的 ```cypher 标记
        clean_cypher = cypher_code.replace("```cypher", "").replace("```", "").strip()
        
        result_data = []
        error_message = None
        
        try:
            with self.driver.session() as session:
                # 执行查询
                result = session.run(clean_cypher)
                # 转换结果为列表字典格式，方便后续展示
                for record in result:
                    result_data.append(record.data())
                    
            return {
                "status": "success",
                "data": result_data,
                "error": None
            }

        except Exception as e:
            # --- 关键：捕获报错堆栈 ---
            # 获取详细的错误描述，这是传给 Agent 反思的“核心素材”
            full_error = traceback.format_exc()
            
            # 简化报错信息，提取最直观的错误原因
            if isinstance(e, CypherSyntaxError):
                short_error = f"语法错误: {str(e)}"
            else:
                short_error = f"运行时错误: {str(e)}"
                
            print(f"⚠️  检测到 Cypher 执行异常: {short_error}")
            
            return {
                "status": "error",
                "data": [],
                "error": short_error,
                "full_traceback": full_error # 也可以选择把完整的堆栈发给模型
            }

# --- 简单测试脚本 ---
if __name__ == "__main__":
    executor = Neo4jExecutor()
    # 模拟一个错误的查询
    test_res = executor.run_query("MATCH (n:Sensor) RETUR n LIMIT 5") # 故意少写个 N
    print(f"执行状态: {test_res['status']}")
    print(f"捕获到的错误: {test_res['error']}")
    executor.close()