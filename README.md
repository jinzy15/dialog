# dialog
to find good answer

### 可以加强的地方
#### 过滤
保留英文，连接等专有名词
addrank的set可以不是只用最后一轮对话进行选取，可以在全数据集上选取


### 每种distance需要使用的数据格式
#### AddRank
1.所有的对话
2.分词
3.过滤（过滤不好应该会变成unk）
4.将一个人说的话进行融合
5.将少于一个对话的（没有回答进行删除）
6.不要忘记进行查询的对话要没有回答（最后一句话为客户提出的问题，要进行融合）
7.需要把所有的
#### BaseRank
与addrank相同
#### bleuRank
暂时和addrank相同
#### 



