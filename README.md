# 2020智慧海洋建设
2020数字中国创新大赛-数字政府赛道-智能算法赛：智慧海洋建设算法赛道B榜 top5代码

赛题链接：https://tianchi.aliyun.com/competition/entrance/231768/introduction

整体采用lightgbm模型5折，最终取得算法赛道复赛B榜 top5

## 运行代码

```shell
> sh run.sh
```


## 文件说明
- run.sh 执行代码
- Dockerfile可忽略，是线上提交的需要

- nmf_list.py是用做tfidf的特征处理

- feature_selector是用做特征选择

## 模型大致思路

- 将所有数据data切分成：速度等于0和非0、白天和黑夜，四个数据集对每艘船的速度、方向、xy进行
统计。

- 采用**tfidf**对速度和xy进行抽取特征并降维

- 采用**自然语言思路**对速度、xy进行embedding

- 训练模型前采用**Lightgbm**进行初步的特征筛选

- 最后用**Lightgbm**进行模型训练

队伍成员天池id：大白_、jycoco、=CODE.KITTY=、挽着我衣袖的姑娘n_
