from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain.prompts import (
    PromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
load_dotenv()
# 创建Neo4jGraph实例并刷新schema
graph = Neo4jGraph()
graph.refresh_schema()

# 设置对话上下文的内存
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 定义系统提示
system_prompt = """
任务：生成查询图数据库的 Cypher 语句。
指令：只使用提供的关系类型和属性进行查询。不要使用任何未提供的关系类型或属性。
模式：{schema}

生成指令步骤：
1. 根据对话上下文和当前问题，生成最佳的当前指令。
2. 基于生成的当前指令，构建相应的 Cypher 查询语句。
{chat_history}

指令生成的结果要求：
1. 不要在你的响应中包含任何解释或道歉。
2. 不要回答任何要求你构建 Cypher 语句之外的问题。
3. 除了生成的 Cypher 语句，不要包含任何文本。
4. 必须生成有效的Cypher代码。即使用户没有问问题也必须生成正确的Cypher代码。

重要提示：
1. 带有 `参展公司` 标签的节点不会包含 `name` 属性，请勿在查询中使用此属性。
2. 带有 `参展公司` 标签的节点中，有些包含`上下游合作伙伴` 属性，有些不包含，请在查询中考虑这一点。
3. 使用 WHERE n.`属性` CONTAINS '值' 形式的查询语句，优先级高于 WHERE n.`属性` = '值'。
4. `会馆` 标签的节点下方的关系只有 `第一层` 和 `第二层`，不要使用其他关系类型。
5. Cypher 语句中不能直接使用 `UNION` 将结果集连接起来。
6. `UNION` 必须连接具有相同列名和类型的结果集。在返回结果中，`RETURN` 子句中的属性名必须完全一致。
7. 忽略 `馆内静态公共设施`这一个标签与其相关的所有内容。 
8. 在生成Cypher语句的时候不要出现EXISTS相关的代码。

以下是一些针对特定问题生成的 Cypher 语句示例：

1. 来自德国的参展公司有哪些？
MATCH (n:`参展公司`)
    WHERE n.`国家_地区` CONTAINS '德国'
RETURN n.公司名称, n.外语名称, n.基本信息, n.主要展品类型, n.展区位置

2. 来自印度的参展公司有哪些？
MATCH (n:`参展公司`)
    WHERE n.`国家_地区` = '印度'
RETURN n.公司名称, n.外语名称, n.基本信息, n.主要展品类型, n.展区位置

3. 会馆3内部有哪些公司？
MATCH (h:`会馆`)-[:第一层|第二层]->(n:`参展公司`) 
RETURN n.上下游合作伙伴 , n.外语名称 , n.基本信息 , n.主要展品类型 , n.公司名称 

4. 印度参展的迈大集团和瑞迪博士这两家公司的展位分别在哪里？
MATCH (h:`会馆`)-[r:第一层|第二层]->(n:`参展公司`)
WHERE n.公司名称 IN ['迈大集团', '瑞迪博士']
RETURN n.展区位置, n.外语名称, n.公司名称, n.国家_地区, h.name AS 会馆名称, 
       CASE 
           WHEN type(r) = '第一层' THEN '第一层' 
           WHEN type(r) = '第二层' THEN '第二层' 
       END AS 展馆楼层

5. 进博士有哪些功能？
MATCH (n:`进博士`)-[:应用模块]->(m)
RETURN DISTINCT m.`服务名称` AS 功能, m.`服务介绍`

6. 把***公司的所有信息都给我列出来
MATCH (n:`参展公司`)
WHERE n.`公司名称` CONTAINS '***'
RETURN n

7. 我想吃饭了/我饿了/有什么推荐的餐馆...（这一类想要吃正餐或者小吃的要求）
MATCH (n:`商业广场商户信息`)
WHERE n.`商铺类别` CONTAINS '餐' or n.`商铺类别` CONTAINS '店' or n.`商铺类别` CONTAINS '吃'
RETURN n

8. 我想喝点水/渴了/哪里有卖水的...（这一类想要喝饮品的要求）
MATCH (n:`商业广场商户信息`)
WHERE n.`商铺类别` CONTAINS '咖啡' or n.`商铺类别` CONTAINS '店'
RETURN n

9. ***店在哪里
MATCH (n)
WHERE n.`品牌名称` CONTAINS '***' or n.`公司名称` CONTAINS '***'
RETURN n
问题是：{question}
"""

answer_prompt = '''
你是一个助手，帮助形成友好且易于理解的答案。信息部分包含你必须用来构建答案的提供信息。
提供的信息是权威的，你绝不能怀疑或试图使用你的内部知识来纠正它。
让答案听起来像是对问题的回应。
不要提及你是基于给定信息得出结果的。
同时结合上下文和当前的指令给出优化过的回答
下面是一个例子：

问题：哪些经理拥有Neo4j股票？
上下文：[经理：CTL LLC，经理：JANE STREET GROUP LLC]
有用的回答：CTL LLC，JANE STREET GROUP LLC拥有Neo4j股票。

在生成答案时，请参考这个例子。
如果提供的信息为空，请你先说没有检索到您说的内容然后结合上下文后再来回复。
信息：
{context}

问题：{question}
有用的回答：
'''


CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=system_prompt
)
QA_GENERATION_PROMPT = PromptTemplate(
    input_variables=['context', 'question'],
    template=answer_prompt
)
# 创建LLM实例
llm = ChatOpenAI(
    temperature=1,  # 设置较低的温度以提高生成结果的一致性
    model="glm-4",
    openai_api_key="your_api_key/你自己的api密钥",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 创建 GraphCypherQAChain 实例
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    memory=memory,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=QA_GENERATION_PROMPT
)

# 开始用户交互循环
def main():
    while True:
        content = input('user:')

        if content == 'quit':
            exit()
        # daye={"query": content, "man": "manbaout"}
        # chain.prep_inputs()
        response=chain.invoke({"query": content})
        # print({chat_history})
        if isinstance(response, dict) and 'result' in response:
            print('answer:', response['result'])
        else:
            print('answer:', response)

main()
