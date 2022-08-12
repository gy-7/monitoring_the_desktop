# MMdeploy TensorRT 模型实时监控桌面，PyQt5实现



:star::star::star::star::star:  本项目遵从：[GNU General Public License v3.0](https://github.com/gy-7/monitoring_the_desktop/blob/main/LICENSE) 



## 个人博客『 gy77 』：

- [x] [GitHub仓库](https://github.com/gy-7/monitoring_the_desktop) ：代码源码详见仓库 demo_qt.py
- [x] [我的CSDN博客](https://blog.csdn.net/qq_39435411)  
- [x] [我的博客园](https://www.cnblogs.com/gy77/) 



## 简介：

利用PyQt5搭建界面，使用mmdeploy的api，加载转换好的TensorRT模型，监控桌面。

分享一下我导出的yolox-s TensorRT模型（RTX 2060s，RTX 2060可用，其余的20系没测试，10系，30系显卡用不了）：[yolox-s TensorRT模型](https://wws.lanzouw.com/i6nfv09f8d5g) 

如果链接挂了，可以发邮件给我：gaoying2020@163.com 



所用python包：

```powershell
python: 3.7
PyQt5: 5.15.7
mmdeploy_python: 0.5.0 
opencv-python: 4.5.5.62
numpy: 1.21.6
qimage2ndarray: 1.9.0
```



## 界面截图：

#### 启动界面：

![1](https://gy77-blog.oss-cn-hangzhou.aliyuncs.com/img/1.jpg)



#### 推理图片：

![2](https://gy77-blog.oss-cn-hangzhou.aliyuncs.com/img/2.jpg)



#### 监控桌面：

![3](https://gy77-blog.oss-cn-hangzhou.aliyuncs.com/img/3.jpg)



![4](https://gy77-blog.oss-cn-hangzhou.aliyuncs.com/img/4.jpg)
