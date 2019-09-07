## 文档图片数据生成（基于图片处理）
### 目录结构
```
- config: 配置文件
- data: 数据
    - all_char_imgs: 字符图片（宋体，汉字）
    - all_char_imgs3: 字符图片（宋体，数字字母标点等其他字符）
    - font2: 黑体
    - all_chars(3).txt: corpus.json包含的汉字（字母数字标点等）
    - corpus: 语料，搜狗新闻
    - search_dict(3).json: 汉字（字母数字标点）搜索字典，字符－单字图片路径
    - search_dict_all.json: 汉字字母数字标点等并集
- generation: 数据生成脚本
- gen_char_imgs.py: 单字字符图片生成脚本
    - lines_generation.py: 根据单字生成文本行
    - paragraphs_generation: 数据生成主文件
    - pick_valid_corpus.py: 抽取符合要求的语料
    - insert_images.py: 插入插图
    - post-processing: 后处理，旋转添加噪声等
```

### 依赖
opencv-python, numpy, yaml等

### 运行
1. 执行gen_char_imgs.py脚本，生成单字图片，可以指定字体类型与保存路径
2. 根据需要修改config/conf.yaml中的数据生成配置
3. 运行generation/paragraphs_generation.py生成合成数据