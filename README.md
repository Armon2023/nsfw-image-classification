# **nsfw**

项目文件压缩包：http://speedtail.fun/download/nsfw-image-classification.zip

### introduction

- 基于tensorflow Inception V3模型迁移学习的图片鉴定程序
- 色情图片鉴定
- 血腥图片鉴定

  代码是在官网的代码基础上更改后的,由于python水平有限,因此比较乱.
  图片分类的最终正确率在95%左右,该模型在1W张图片之内就达到了极限,过多次的重复训练会导致过拟合.

### Required Packages

此程序运行需要安装几个Python包。这些包及版本如下(目前代码中使用的是cpu版本的tensorflow):

- tensorflow           1.13.1
- numpy                 1.16.2
- Python                 3.5.0
- tornado                5.1.1

安装时最好对应版本,否则会有很多坑(py大神随意吧!)

### Running the model

命令行进入对应主目录输入

`python ./label_image.py`

启动成功后在浏览器输入

http://127.0.0.1:{port}/porn?image={imageUrl}  ||  http://127.0.0.1:{port}/bloody?image={imageUrl} 

可以看到输出的json概率
