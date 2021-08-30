pytorch ==1.7.0
cuda ==11
cudnn ==8
Pillow==7.2.0
path==15.0.1
imutils==0.5.3


测试程序  inference.py
输入  128行   source = 'image_test/image/'  可以是文件夹路径（自动读取文件夹下所有图片）也可以是一张图片

输出 349行 dstFileName = 'image_test/test/'+str(cout)+'.jpg'  保存的路径 已经图片名称