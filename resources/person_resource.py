import sqlite3
import os
from resources import YA_MCPServer_Resource

# Add a person vector to the database
@YA_MCPServer_Resource(
    "db:///core/database/person_recognition.db",
    name="person_recognition_database",
    title="Person Recognition Database",
    description="人物特征向量数据库获取人物特征",
)
def get_database() -> sqlite3.Connection:
    """
    查询人物特征向量从数据库
    Args:   
    Returns:
        sqlite3.Connection: 数据库连接对象
    """
    try:
        db_path = os.path.join(os.path.dirname(__file__), "../core/database/person_recognition.db")
        return sqlite3.connect(db_path)
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to connect to database: {e}")



