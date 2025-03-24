from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse, urlunparse, quote, unquote
import pymysql
from dify_plugin import Tool
from dify_plugin.entities.model.llm import LLMModelConfig
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from pymysql.cursors import DictCursor
import re

class RookieText2dataTool(Tool):
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        model_info= tool_parameters.get('model')
        meta_data = tool_parameters.get('meta_data')
        # 获取连接参数
        _, conn_params = self._build_mysql_dsn(
            self.runtime.credentials['db_url'],
            self.runtime.credentials['db_password']
        )
        # 获取元数据
        # metadata = self._get_metadata(conn_params)

        # 给深度思考模型的两阶段系统提示
        system_prompt = f"""你是一位资深数据库工程师兼SQL优化专家，拥有10年以上DBA经验。
我将提供数据库元数据和一个用自然语言表达的查询需求，请你生成优化的SQL查询语句。

请用以下两步完成：

第一步：分析用户需求并理解数据结构
- 深入理解用户的自然语言查询意图
- 识别查询中涉及的实体、关系和条件
- 确认哪些表和字段需要使用

第二步：构建SQL语句并优化
- 根据第一步分析构建基础SQL语句
- 应用以下优化和安全规则：
  1. 仅返回SELECT语句，不包含INSERT/UPDATE/DELETE等DML操作
  2. 使用LIMIT语句限制结果（用户未指定数量时默认100条）
  3. 所有字段使用反引号包裹，符合MySQL标识符规范
  4. 避免SELECT *，仅返回需求中必要字段
  5. 多表关联优先使用INNER JOIN
  6. 确保严格使用提供的元数据中的表和字段，不允许使用未定义内容

数据库元数据：
{meta_data}

返回格式示例：
SELECT 
    `order_id` AS 订单编号,
    `amount` * 1.05 AS 含税金额
FROM 
    `orders` o
INNER JOIN 
    `customers` c ON o.customer_id = c.id
WHERE 
    o.status = 'paid' 
    AND c.region = 'Asia'
LIMIT 100;"""

        response = self.session.model.llm.invoke(
            model_config=LLMModelConfig(
                provider=model_info.get('provider'),
                model=model_info.get('model'),
                mode=model_info.get('mode'),
                completion_params=model_info.get('completion_params')
            ),
            prompt_messages=[
                SystemPromptMessage(content=system_prompt),
                UserPromptMessage(
                    content=f"用户想要查询的数据需求为：{tool_parameters.get('query')}"
                )
            ],
            stream=False
        )

        excute_sql = response.message.content

        # 执行SQL
        result = self._execute_sql_generator(excute_sql, conn_params)

        yield self.create_json_message({
            "status": "success",
            "data": result
        })

    def _build_mysql_dsn(self, db_url: str, db_password: str) -> tuple[str, dict[str, Any]]:
        """
        将数据库URL和密码拼接为完整DSN，并返回解析后的连接参数
        
        参数：
            db_url (str): 格式示例 mysql://user@host:port/database
            db_password (str): 数据库密码（明文）
        
        返回：
            tuple: (完整DSN, 解析后的连接参数字典)
        
        异常：
            ValueError: 当URL格式无效时抛出
        """
        # 基础解析验证
        parsed = urlparse(db_url)
        if parsed.scheme != 'mysql':
            raise ValueError("仅支持mysql协议，当前协议：{}".format(parsed.scheme))

        # 解析用户名和主机信息
        username = parsed.username or 'root'
        password = quote(db_password, safe='')  # 处理特殊字符
        hostname = parsed.hostname or 'localhost'
        port = parsed.port or 3306

        # 处理数据库路径
        database = parsed.path.lstrip('/')
        if not database:
            database = 'test'

        # 构建新的netloc
        auth_part = f"{username}:{password}"
        netloc = f"{auth_part}@{hostname}"
        if parsed.port:
            netloc += f":{port}"

        # 生成完整DSN
        full_dsn = urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

        # 生成连接参数字典（类型安全）
        connection_params = {
            'host': hostname,
            'port': port,
            'user': unquote(username),
            'password': unquote(db_password),  # 注意这里返回明文用于连接
            'database': database,
            'charset': 'utf8mb4',
            'connect_timeout': 5
        }

        return full_dsn, connection_params

    def _get_metadata(self, conn_params: dict[str, Any]) -> dict[str, Any]:
        """
        获取数据库元数据（表结构信息）
        
        返回结构示例：
        {
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {"name": "id", "type": "int", "comment": "主键ID"},
                        {"name": "name", "type": "varchar(255)", "comment": "用户名"}
                    ]
                }
            ]
        }
        """
        metadata = {"tables": []}
        
        try:
            with pymysql.connect(
                host=conn_params['host'],
                port=conn_params['port'],
                user=conn_params['user'],
                password=conn_params['password'],
                database=conn_params['database'],
                charset=conn_params['charset'],
                cursorclass=DictCursor
            ) as conn:
                with conn.cursor() as cursor:
                    # 获取所有表信息
                    cursor.execute("""
                        SELECT TABLE_NAME AS table_name,
                        TABLE_COMMENT AS table_comment
                        FROM INFORMATION_SCHEMA.TABLES
                        WHERE TABLE_SCHEMA = DATABASE()
                        AND TABLE_TYPE = 'BASE TABLE'
                    """)
                    tables = cursor.fetchall()

                    # 获取每个表的列信息
                    for table in tables:
                        cursor.execute("""
                            SELECT COLUMN_NAME AS name,
                                COLUMN_TYPE AS type,
                                COLUMN_COMMENT AS comment,
                                IS_NULLABLE AS nullable,
                                COLUMN_KEY AS key_type
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_SCHEMA = DATABASE()
                            AND TABLE_NAME = %s
                            ORDER BY ORDINAL_POSITION
                        """, (table['table_name'],))
                        
                        columns = []
                        for col in cursor.fetchall():
                            columns.append({
                                "name": col['name'],
                                "type": col['type'],
                                "comment": col['comment'] or "",
                                "nullable": col['nullable'] == 'YES',
                                "primary_key": col['key_type'] == 'PRI'
                            })
                        
                        metadata['tables'].append({
                            "name": table['table_name'],
                            "comment": table['table_comment'] or "",
                            "columns": columns
                        })
        
        except pymysql.Error as e:
            code, msg = e.args
            error_map = {
                1142: ("权限不足，无法访问元数据表", 403),
                1045: ("数据库认证失败", 401),
                2003: ("无法连接数据库服务器", 503)
            }
            error_info = error_map.get(code, (f"数据库错误: {msg}", 500))
            raise RuntimeError(f"{error_info[0]} (错误码: {code})") from e
            
        return metadata
    
    def _execute_sql_generator(self,sql: str, conn_params: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        """
        基于PyMySQL的SQL执行生成器函数
        :param sql: 待执行的SQL语句
        :param conn_params: 数据库连接参数字典
        :yield: 返回包含执行状态和数据的字典
        """
        extracted_sql = self._extract_sql_from_text(sql)
        if not extracted_sql:
            # 如果无法提取（未匹配到代码块），直接使用原始字符串
            processed_sql = sql.strip()
        else:
            processed_sql = extracted_sql
        
        connection = None
        try:
            # 建立数据库连接
            connection = pymysql.connect(
                host=conn_params['host'],
                user=conn_params['user'],
                password=conn_params['password'],
                database=conn_params['database'],
                charset='utf8mb4',  # 必须指定字符集
                cursorclass=DictCursor  # 返回字典类型结果
            )
            with connection.cursor() as cursor:
                # 执行SQL语句
                cursor.execute(processed_sql)
                
                # 获取列名和数据（引用[2,5](@ref)）
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                
                # 生成结果集
                yield {
                    "status": "success",
                    "sql": processed_sql,
                    "columns": columns,
                    "data": results
                }
                
        except pymysql.MySQLError as e:
            # 捕获数据库特定错误
            yield {
                "status": "error",
                "excute_sql": processed_sql,
                "message": f"Database error: {str(e)}"
            }
        except Exception as ex:
            # 捕获其他异常
            yield {
                "status": "error",
                "excute_sql": processed_sql,
                "message": f"Execution error: {str(ex)}"
            }
        finally:
            # 资源释放
            if connection and connection.open:
                connection.close()
                yield {
                    "status": "info",
                    "message": "数据库连接已关闭"
                }

    def _extract_sql_from_text(self, text: str) -> str:
        """提取SQL内容，支持多种格式输出（包括深度思考模型的分析输出）"""
        # 先尝试匹配代码块中的SQL
        code_block_patterns = [
            r'```sql\s*(.*?)\s*```',  # SQL标准代码块
            r'```\s*(SELECT.*?LIMIT.*?)\s*```',  # 无语言标记但包含SELECT和LIMIT的代码块
            r'`{3}(SELECT.*?LIMIT.*?)`{3}'  # 另一种代码块形式
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 尝试直接匹配SELECT语句
        select_patterns = [
            r'(SELECT\s+.*?LIMIT\s+\d+\s*;)',  # 带分号的完整SELECT语句
            r'(SELECT\s+.*?LIMIT\s+\d+)'       # 不带分号的SELECT语句
        ]
        
        for pattern in select_patterns:
            match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 处理深度思考模型可能输出的分析+SQL组合
        # 先尝试切分，找到最后部分的SQL
        lines = text.split('\n')
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip().upper().startswith('SELECT'):
                # 从这一行开始提取到结尾
                potential_sql = '\n'.join(lines[i:])
                # 如果包含LIMIT则返回
                if 'LIMIT' in potential_sql.upper():
                    return potential_sql.strip()
        
        # 如果所有尝试都失败，返回原始文本
        return text.strip()