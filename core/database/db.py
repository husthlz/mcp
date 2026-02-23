import sqlite3
import logging
from typing import Tuple
import json

logger = logging.getLogger(__name__)

# Initialize the database
def init_db(db_path: str = 'core/database/person_recognition.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            feature_vector TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    
    
def save_person_features(person_name: str, features: list, db_path: str = 'core/database/person_recognition.db') -> bool:
    """
    将提取的人物特征存储到数据库

    Args:
        person_name (str): 人名
        features (list): 人物特征向量
        db_path (str): 数据库路径

    Returns:
        bool: 存储成功返回True，否则返回False
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 确保表存在
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                feature_vector TEXT NOT NULL
            )
        ''')

        features_str = json.dumps(features)  # 将特征列表转换为JSON字符串

        # 插入数据
        cursor.execute(
            'INSERT INTO person_vectors (person_name, feature_vector) VALUES (?, ?)',
            (person_name, features_str)
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"存储人物特征时出错: {str(e)}")
        return False
    
def update_person_features(person_name: str, features:list, db_path: str = 'core/database/person_recognition.db') -> bool:
    """
    更新数据库中已存在的人物特征

    Args:
        person_name (str): 人名
        features (list): 人物特征向量
        db_path (str): 数据库路径

    Returns:
        bool: 更新成功返回True，否则返回False
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        features_str = json.dumps(features)  # 将特征列表转换为JSON字符串

        # 更新数据
        cursor.execute(
            'UPDATE person_vectors SET feature_vector = ? WHERE person_name = ?',
            (features_str, person_name)
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"更新人物特征时出错: {str(e)}")
        return False

def query_person_vectors(person_name: str, db_path: str = 'core/database/person_recognition.db') -> Tuple[int, str, list]:
    '''
    查询人物特征向量从数据库
    
    Args:
        person_name (str): 人名
        db_path (str): 数据库路径
        
    Returns:
        Tuple[int, str, list]: 包含ID、姓名和特征向量的元组，未找到返回None
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM person_vectors WHERE person_name = ?', (person_name,))
    row = cursor.fetchone()  # 获取第一条记录
    conn.close()
    if row:
        feature_list = json.loads(row[2])  # 将JSON字符串转换回列表
        return (row[0], row[1], feature_list)
    else:
        return None