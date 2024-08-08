import pandas as pd
from neo4j import GraphDatabase

class ExpoGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        # 处理属性名中的空格和特殊字符
        properties = {k.replace(" ", "_").replace("（", "").replace("）", "").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_").replace("、", "_"): v for k, v in properties.items()}
        query = f"CREATE (n:{label} {{"
        # 使用repr()函数来对属性值进行转义
        query += ", ".join([f"`{k}`: {repr(v)}" for k, v in properties.items()])
        query += "}) RETURN n"
        # print(query)
        self.query(query)

    def query(self, query):
        with self.driver.session() as session:
            session.run(query)

# 连接到Neo4j
graph = ExpoGraph("bolt://localhost:7687/", "neo4j", "whatcanisay")

# 读取Excel文件
file_path = './导入.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# 过滤无效列（如Unnamed列）
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# 将数据插入到Neo4j中
for index, row in data.iterrows():
    properties = row.to_dict()
    # 将NaN值转换为空字符串
    properties = {k: (v if pd.notna(v) else "") for k, v in properties.items()}
    graph.create_node("参展公司", properties)
graph.close()
