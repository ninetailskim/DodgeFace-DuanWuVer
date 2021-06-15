## 利用Paddlehub制作端午体感小游戏

### 前言
上次直播的最后和大家简单聊了一下换皮.因为直播讲的和github上release上有一些差别(直播讲了一些打算要改进的地方),所以想着干脆重构一下吧.   
恰巧碰上了端午节,所以干脆再重写一些逻辑,做个端午节定制小游戏吧.   

#### 你可以在这里看到游戏哦   
<iframe style="width:98%;height: 800px;" src="//player.bilibili.com/player.html?aid=888624921&bvid=BV1UK4y137AF&cid=353600257&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

### 端午特色   
游戏的贴图全换成了端午节相关贴图:三种粽子造型   
![](https://ai-studio-static-online.cdn.bcebos.com/83db99973f8a4f0789370c13107b6b47dc6e6399910740cb84e15c6470a8835a)
![](https://ai-studio-static-online.cdn.bcebos.com/fc7844abc36d4feb8727ba799ca8debfbf58405ce1554a0799eed07491d4506c)
![](https://ai-studio-static-online.cdn.bcebos.com/09a549e0a8444a538c6bad164dddc499aef1f1b6a5374063b7e3d29e89f82e7a)   
雄黄酒   
![](https://ai-studio-static-online.cdn.bcebos.com/33971c67e93941a0b7e10cccabb8778c8e6cb2bb6e2441ac93f91a91c6aa1438)   
以及五毒:蛇,壁虎,蜈蚣,蟾蜍,蟹子   
![](https://ai-studio-static-online.cdn.bcebos.com/bcdd44637a1347bf9ec99fb4a2256066750ca3ae10ca4a0a9a85d7725236ca74)
![](https://ai-studio-static-online.cdn.bcebos.com/f670fa3f219e4c7d89c6712d6d70a2c3604c484336164ba89a03053a380a2c46)
![](https://ai-studio-static-online.cdn.bcebos.com/322bfece50ff45efbe398bf4a1c8f45eafc8e201358b409cbfd1fd68d60cf171)
![](https://ai-studio-static-online.cdn.bcebos.com/5915ea7102f5482faff34a9c069ef95a51c0ae3fd17548c194f40dddb3c93f8b)
![](https://ai-studio-static-online.cdn.bcebos.com/03dabbea7fea418f9261cff15baa32812bedf7cebd9f417398b078df7dd481b6)   

其实五毒也是我在逛了粽子博物馆才看到的哈哈哈,所以虽然做的是个游戏,但是也是有一些科普的性质在里面的.   
其实除了避五毒,端午节还要吃五黄,分别是:黄鳝,黄鱼,黄瓜,黄泥蛋以及雄黄酒.   
故人认为端午仲夏,"五毒"大量繁殖,易咬伤人,吃了"五黄"能够抵御"五毒"的侵害.   
是不是觉得又掌握了一些小知识呢~   

### 运行游戏
由于需要摄像头,所以无法在aistudio上运行,请clone github仓库到本地运行.
```python duanwu.py --level(optional)```

### 游戏说明
改变了之前可以随便移动的控制方式,这次控制的小熊只能在屏幕的最低端左右移动.根据摄像头的图像,映射出X轴的位置即可.   
游戏一共100秒.   
吃到粽子会加100分.   
吃到雄黄酒会进行一次清屏一次性加1000分,但是喝酒后会左右颠倒进入眩晕状态,要注意操作方式.在眩晕的期间,吃到粽子分值翻倍.   
吃到五毒的话......会直接结束游戏,哈哈,要注意喽   

### 改进   
在直播中给大家讲了做游戏的一中单例设计模式,Manager的概念.(当然,这不是一种很好的方式,但是做这种简单的小游戏已经完全足够)   
这次代码重构,则完全按照直播的规划来重构的.重新设计了UIManager, ResourceManager, GameManager, BallManager等等.代码的可读性更高更简洁,耦合性则更低.   