
# 项目背景
本仓库是datafoundation的[多模态客服智能体竞赛](https://www.datafountain.cn/competitions/1165/ranking?isRedance=0&sch=2580&stage=B)的实现方案，A榜第0.855分，第十名，B榜0.83875分，第七名。
该赛题需要参赛者构建一个多模态的客服智能体，针对用户问题，精准返回手册中对应的内容和相关配图。赛题中给了问题包含两类，一种为通识类，一种产品类。通识类问题并没有相关的知识库进行检索，需要模型理解问题并回答，产品类问题则需要参赛者构建一个RAG系统，先利用所给的中英文手册构建知识库，之后针对用户问题进行检索，最终智能体需要返回手册中对应的内容以及配图，图文互补，提升用户的理解效果。

# 整体架构图
> 完整的技术方案查看[技术方案.md](技术方案.md)


1. **离线处理阶段：**首先利用LLM对原手册中的内容进行预处理，再利用LLM将手册中的插图转为自然语言描述并替换原来手册中的`<PIC>`标签，之后我们根据处理后的手册中的markdown标题层级进行切分并向量化存入向量数据库中。

2. **问答阶段：**我们设计了一个**多层的查询分类方法**，首先识别用户的语言:英文还是中文，再将用户问题分为通识类和产品类，并且针对产品类问题进一步找出是针对哪个产品进行提问（即判断在哪个产品手册中可以找到答案）

   - 通识类问题由于不需要进行检索，我们设计了提示词，并测试了不同模型的效果，最终采用了`gemini-3-flash-preview`or `gemini-3.5-flash`进行回答（二者效果基本一致）
   - 产品类问题：我们根据上面得到的语言和产品手册文档信息作为先验知识来缩小向量检索范围，最后采用向量检索+BM25检索的混合检索方式来检索出最后的top19的片段，并按原文档的顺序进行拼接起来最后给LLM进行回答。最后我们再使用LLM对回答进一步细化以提升最终的答案效果。
![整体架构图-非手绘](https://img.leftover.cn/img-md/202606201328116.png)


# 项目目录说明
```python
cs_agent
├─ README.md
├─ answer_general_query.py #处理通识类问题的代码
├─ answer_product_query.py # 处理产品类问题的代码
├─ catalog # 分别对中英文手册生成的目录内容
│  ├─ chinese_handbook_catalog.json
│  └─ english_handbook_catalog.json
├─ chunk.py # 手册切块、入库的代码
├─ config.py # 配置文件，只需配置MILVUS_COLLECTION_NAME_DEFAULT变量即可，设置向量数据库的集合名称
├─ data # 原始数据
├─ del_bank_by_handbook.py # 根据手册名删除向量数据库中该手册的所有数据
├─ generate_catalog.py # 生成手册目录的代码
├─ generate_handbook_name.py # 生成英文手册名称的代码
├─ handbook_name_gemini.json # 存放生成的英文手册名称
├─ interface.py # 接口代码，提供给前端调用
├─ llm_judge_result.py # 使用LLM给最终的结果打分的代码
├─ milvus-docker-compose.yml # 部署milvus向量数据库的docker-compose文件
├─ pipline.py #  智能体的入口文件
├─ preprocess.py # 预处理代码，使用LLM对中英文手册内容进行预处理
├─ processed_data # 预处理之后的数据，其中_formatted.txt为后缀的文件会用于接下来的切块和入库
├─ prompts.py # 存放一些提示词
├─ pyproject.toml 
├─ query_classification.py # 对问题进行分类，预测手册名称的代码
├─ question_public.csv # 公开的测试问题
├─ retriever.py # 检索代码，根据问题检索相关的手册内容
├─ submission # 存放提交结果的文件夹
├─ submit.py # 根据测试问题生成运行pipeline，生成最终的提交结果
├─ query_classification_results.json # 查询分类的测试结果
├─ utils.py # 工具函数
|- milvus-backup-files #向量数据库的备份
|- backup.yaml #恢复备份的时候需要用到的配置文件
|- mc #minio的命令行工具
|- milvus-backup # milvus的备份和恢复的工具
└─ uv.lock
```



# 依赖安装

1. 下载uv

   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. 进行项目根目录，执行下面的命令下载依赖

   ```shell
   uv sync -i https://pypi.tuna.tsinghua.edu.cn/simple
   source .venv/bin/activate
   ```

3. 配置key,按文件中的注释要求配置即可

   ```shell
   cp .env.example .env
   ```

4. 配置.env 和 config.py中的内容，config.py中只要配置MILVUS_COLLECTION_NAME_DEFAULT变量即可，设置向量数据库的集合名称

5. 启动milvus数据库，

   ```shell
   sudo docker compose -f milvus-docker-compose.yml up -d
   ```

# 使用已经有的向量数据库复现

1. 可以选择重新建立向量数据库，也可以使用已经建立好的向量数据库

   重新建立向量数据库，需要运行`chunk.py`,对预处理好的数据进行切块并将pic转为对应的描述，再入库

   如果需要使用我建立的向量数据库，则需要在启动之后将数据导入进去，`milvus-backup-files`中就是对应的数据,只需要在启动了milvus数据库了之后运行下面的命令即可将数据导入向量数据库中即可

```shell
./mc alias set dst http://127.0.0.1:9000 minioadmin minioadmin
./mc mb dst/a-bucket --ignore-existing
./mc cp --recursive \
  ./milvus-backup-files/milvus_to_aliyun_20260615 \
  dst/a-bucket/backup/
  
./milvus-backup restore \
  -n milvus_to_aliyun_20260615 \
  --config backup.yaml
```

2. 启动接口

   ```shell
   python interface.py
   ```

   访问`http://localhost:8000/scalar` 查看接口文档



# 从0开始复现

1. 运行`preprocess.py`使用`gemini-2.5-pro`对手册预处理

2. 之后运行`chunk.py`对文本进行切块、将pic转为文本描述，最后将chunk向量化并入库

3. 运行`generate_handbook_name.py`生成英文的手册名称

4. 运行`generate_catalog.py`生成手册的目录

5. 启动接口

   ```shell
   python interface.py
   ```

   访问`http://localhost:8000/scalar` 查看接口文档


# 性能、成本分析
## 性能分析

我们的智能体在90%的情况下，查询分类只需要调用一次LLM，最多调用两次LLM。

- 通识性问题在查询分类结束后，LLM直接处理。

  **因此对于通识性问题最少调用两次LLM，最多调用三次。**

- 产品类问题在查询分类结束后会进行混合检索，由于我们先对问题进行了分类，因此我们可以在检索阶段先进行过滤，从而不需要昂贵的rerank操作就可以达到很好的检索效果。接下来会使用LLM对其生成答案，再使用LLM对答案进行优化。

  **因此对于产品类问题，90%的情况下只需要调用三次LLM，一次embedding模型**

  当查询分类有误时，检索出来的答案不对，模型不能回答的时候，我们还设计了一个兜底的方案。会重新进行检索再回答。这会增加**一次embedding模型的调用，一次LLM 的调用**

	我们的方案在大多数情况下可以实现较好的成本**（避免了昂贵的rerank操作）**和时间的控制**（最少只需要调用三次LLM）**，在极端情况下也有兜底的处理方案，尽管这会增加整体的时间。



## 成本分析

- 对于通识类问题，我们的方案直接使用`gemini-3.5-flash`进行回答，几乎不怎么消耗token

> 对`请问你们家的商品支持7天无理由退换货吗？`这个问题进行测试，我们的token消耗情况为：输入token数为1085，输出token数为600，具体token消耗如下图所示：

![Snipaste_2026-06-20_19-27-25](https://img.leftover.cn/img-md/202606221954726.png)

- 对于产品类问题，查询分类这个模块消耗token数量比较少，主要的token消耗绝大部分来源于检索到的上下文，而我们的top_k=19,实际的检索出来的内容不会特别多，在一个可接受的范围，并且我们只有两次的LLM调用(答案的生成和答案的优化)使用到了检索的上下文，因此成本是很低的。对于落地来说在一个可接受的范围

> 对 `我想更换健身追踪器的表带，有其他尺寸可选吗？`这个问题进行测试，除去embedding 模型，我们的token消耗情况为：输入token数为9553，输出token为1095，具体的token消耗情况如下图所示：



![Snipaste_2026-06-20_19-26-09](https://img.leftover.cn/img-md/202606221954531.png)
