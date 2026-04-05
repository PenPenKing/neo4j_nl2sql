neo4j_nl2cypher/
├── data/
│   ├── schema.json          # 导出的图谱本体 (Nodes, Rels, Props)
│   └── few_shot.json        # 1000+ 条微调数据的精选样本 (用于RAG召回)
├── core/
│   ├── retriever.py         # 核心：双路召回逻辑 (AC自动机 + 向量检索)
│   ├── processor.py         # 核心：LangChain 运行链 (Prompt + LLM)
│   └── executor.py          # 核心：Neo4j 连接与错误捕获 (Error Traceback)
├── app.py                   # FastAPI 入口 (提供 Web API 接口)
├── demo_cli.py              # 演示脚本：直接在终端看到反思纠错过程
├── Modelfile                # Ollama 模型构建配置文件
└── .env                     # 环境变量 (NEO4J_URL, OLLAMA_HOST等)