from prompts import YA_MCPServer_Prompt

@YA_MCPServer_Prompt(
    name="generate_yolo_prompt",
    title="YOLO Prompt",
    description="生成一个用于调用YOLO算法的prompts",
)
def generate_yolo_prompt(task_type: str, image_path: str, conf: float = 0.5) -> str:
    """
    生成用于调用YOLO算法的MCP prompts。

    参数：
        task_type (str): 任务。
        image_path (str): 输入图像的路径。
        conf (float): 置信度阈值，默认为 0.5。

    返回：
        str: 生成的MCP prompts。
    """

    prompt = (
        f"使用YOLO算法执行{task_type}任务。\n"
        f"输入图像路径：{image_path}\n"
        f"置信度阈值：{conf}\n"
        f"请调用对应的工具函数，例如 yolo_{task_type}。"
    )
    return prompt

@YA_MCPServer_Prompt(
    name="count_person_appearances_prompt",
    title="Count Person Appearances",
    description="生成一个用于统计目标人物在图片集中出现次数的prompts",
)
def image_appearances_prompt(target_image: str, image_folder: str) -> str:
    """
    生成用于调用工具统计目标人物在图片集中出现次数的MCP prompts。

    参数：
        target_image (str): 目标人物图片路径。
        image_folder (str): 图片集文件夹路径。

    返回：
        str: 生成的MCP prompts。
    """

    prompt = (
        f"统计目标人物在图片集中出现的次数。\n"
        f"目标人物图片路径：{target_image}\n"
        f"图片集文件夹路径：{image_folder}\n"
        f"请调用tools\\cnt_tool.py中的相关函数完成统计。"
    )
    return prompt

@YA_MCPServer_Prompt(
    name="extract_and_insert_person_features_prompt",
    title="Extract and Insert Person Features",
    description="生成一个用于提取目标人物特征向量并插入数据库的prompts",
)
def extract_features_prompt(target_image: str, person_name: str) -> str:
    """
    生成用于调用工具提取目标人物特征向量并插入数据库的MCP prompts。

    参数：
        target_image (str): 目标人物图片路径。
        person_name (str): 目标人物的名字。

    返回：
        str: 生成的MCP prompts。
    """

    prompt = (
        f"提取目标人物的特征向量并插入数据库。\n"
        f"目标人物图片路径：{target_image}\n"
        f"目标人物名字：{person_name}\n"
    )
    return prompt

@YA_MCPServer_Prompt(
    name="retrieve_and_count_person_appearances_prompt",
    title="Retrieve and Count Person Appearances",
    description="生成一个用于从数据库中提取目标人物特征向量并统计其在图片集中出现次数的prompts",
)
def retrieve_and_count_person_appearances_prompt(person_name: str, image_folder: str) -> str:
    """
    生成用于调用工具从数据库中提取目标人物特征向量并统计其在图片集中出现次数的MCP prompts。

    参数：
        person_name (str): 目标人物的名字。
        image_folder (str): 图片集文件夹路径。

    返回：
        str: 生成的MCP prompts。
    """

    prompt = (
        f"从数据库中提取目标人物的特征向量并统计其在图片集中出现的次数。\n"
        f"目标人物名字：{person_name}\n"
        f"图片集文件夹路径：{image_folder}\n"
        f"请连接数据库提取特征向量，并调用相关工具完成统计。"
    )
    return prompt