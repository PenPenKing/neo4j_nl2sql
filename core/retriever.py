import ahocorasick
import os

class SchemaRetriever:
    def __init__(self, dict_path: str):
        self.ac = ahocorasick.Automaton()
        word_count = 0
        
        # 1. 检查文件是否存在
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        # 只有词典不为空，才添加
                        self.ac.add_word(word, word)
                        word_count += 1
            
            # 2. 核心修复：只有添加了词，才调用 make_automaton
            if word_count > 0:
                self.ac.make_automaton()
            else:
                print(f"⚠️ 警告: 词典文件 {dict_path} 是空的！")
        else:
            print(f"⚠️ 错误: 找不到词典文件 {dict_path}")
            
        self.initialized = word_count > 0

    def get_matched_schema(self, query: str):
        """精准提取问题中的 Schema 关键词"""
        # 如果没初始化成功（没词），直接返回空列表，不跑 iter()
        if not self.initialized:
            return []
            
        results = []
        for end_index, found_word in self.ac.iter(query):
            results.append(found_word)
        return list(set(results))

# --- 测试脚本 ---
if __name__ == "__main__":
    # 确保这里的路径和你实际存放 txt 的位置一致
    # 建议使用绝对路径测试一下，或者确保你在项目根目录下运行
    dict_file = os.path.join(os.path.dirname(__file__), "..", "data", "schema_keywords.txt")
    retriever = SchemaRetriever(dict_file)
    
    test_query = "科学载荷管理器发生了工作电流异常，可能是因为底层的电源故障吗？"
    print(f"🔍 提取到的关键词: {retriever.get_matched_schema(test_query)}")