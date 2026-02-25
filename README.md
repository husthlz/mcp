## YA_MCPServer_re-identification

通过yolo方法与Reid方法实现行人重识别的mcp服务

### 组员信息

| 姓名 | 学号 | 分工 | 备注 |
| :--: | :--: | :--: | :--: |
| 黄律智 | U202414902 | 功能实现与测试 | 无  |
| 刘曜榕 | U202414910 | 模型训练与展示 | 无  |

### Tool 列表

| 工具名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| yolo_tool| 封装yolo模型的基本功能：识别、分割、分类           |  图片路径、选择功能    |  处理后的结果    |  无    |
|person_appearances          | 统计特定人物在图片集中出现的次数         |  特定人物信息    |   统计结果   |  可以通过图片输入或其他数据输入    |
|   db_insert       |     将一条人物信息插入到数据库中     |   人物信息   |   插入结果   |  存储人物特征向量    |
| db_query |从数据库中查询一条人物信息 | 人物名称 | 人物信息 | 包括人物特征向量 | 
| extract_features | 提取图片中的人物特征 | 图片路径 | 提取的人物特征向量 | 无 |

### Resource 列表

| 资源名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| get_database | 连接人物数据库  | 数据库地址  | 数据库连接对象  | 无 |

### Prompts 列表

| 指令名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| generate_yolo_prompt | 生成用于调用YOLO算法的MCP prompts | task_type (str): 任务；image_path (str): 输入图像的路径； conf (float): 置信度阈值，默认为 0.5 | (str)生成的MCP prompts   | 无 |
| image_appearances_prompt |  生成用于调用工具统计目标人物在图片集中出现次数的MCP prompts  | target_image (str): 目标人物图片路径；image_folder (str): 图片集文件夹路径 | (str) 生成的MCP prompts | 无 |
| extract_features_prompt |  生成用于调用工具提取目标人物特征向量并插入数据库的MCP prompts | target_image (str): 目标人物图片路径；person_name (str): 目标人物的名字 |  (str): 生成的MCP prompts | 无 |
| retrieve_and_count_person_appearances_prompt | 生成用于调用工具从数据库中提取目标人物特征向量并统计其在图片集中出现次数的MCP prompts | person_name (str): 目标人物的名字；image_folder (str): 图片集文件夹路径 | (str)： 生成的MCP prompts | 无 |

### 项目结构

- `core`: [核心功能实现]
  - `yolo`:[yolo方法实现]
  - `reid`:[reid方法实现]
  - `database`:[数据库实现]
- `modules`: [模型文件]
  - `yolo_models`:yolo模型文件
- `tools`: [mcp工具]
  - `cnt_tool`:重识别人物出现的统计
  - `db_tool`:数据库操作工具
  - `help_tool`:辅助功能工具
  - `yolo_tool`:封装yolo方法基本功能
- `resources`:[mcp资源]
  - `person_resource`:人物特征数据库 
- `prompts`:[mcp提示词]
- `config.yaml`: [项目配置信息]

### 其他需要说明的情况

- 通过 yolo 模型识别图像中的人物
- 基础框架实现时使用 torchreid 库加载 OSNet 模型，用以提取人物特征向量
