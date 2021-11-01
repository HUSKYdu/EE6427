import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square

import math
import matplotlib.pyplot as plt




def Get_Video(address,num_frame):
    x=[]
    for i in range(0,num_frame):
        h=cv2.imread('./%s/%d.jpg'%(address,(i+1)))
        x.append(h)
    return x
def To_Binary(video):
    videolen=len(video)
    x=[]
    for i in range(0,videolen):
        a=cv2.cvtColor(video[i],cv2.COLOR_BGR2GRAY)
        x.append(a)
    return x
def Get_MSE(video1,video2):
    videolen=len(video1)
    mse=[]
    for i in range(0,videolen):
        a=mean_squared_error(video1[i],video2[i])
        mse.append(a)
    x=np.mean(mse)
    return x
def yuv2bgr(filename, height, width, startfrm,saveaddress):


    fp = open(filename, 'rb')

    framesize = height * width * 3 // 2

    h_h = height // 2

    h_w = width // 2

    fp.seek(0, 2)

    ps = fp.tell()

    numfrm = ps // framesize

    fp.seek(framesize * startfrm, 0)

    for i in range(numfrm - startfrm):

        Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')

        Ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')

        Vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')

        for m in range(height):

            for n in range(width):

                Yt[m, n] = ord(fp.read(1))

        for m in range(h_h):

            for n in range(h_w):

                Ut[m, n] = ord(fp.read(1))

        for m in range(h_h):

            for n in range(h_w):

                Vt[m, n] = ord(fp.read(1))

        img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))

        img = img.reshape((height * 3 // 2, width)).astype('uint8')



        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)

        cv2.imwrite('./%s/%d.jpg'%(saveaddress,(i+1)),bgr_img)

        print("Extract frame %d " % (i + 1))

    fp.close()

    print("job done!")

    return None
#yuv2bgr('football_cif.yuv', height=288, width=352, startfrm=0,saveaddress='0')

# for i in range(0,31):
#     add=i+1
#     yuv2bgr('%d.yuv'%add, height=288, width=352, startfrm=0, saveaddress='%d'%add)
def zhuhanshu():
    video=Get_Video(address='0',num_frame=150)
    video=To_Binary(video)
    mse=[]
    for i in range(0,30):
        print('computing',i+1)
        video1 = Get_Video(address='%d'%(i+1), num_frame=150)
        video1 = To_Binary(video1)
        h=Get_MSE(video,video1)
        mse.append(h)
    return mse
a=zhuhanshu()
print(a)
def Get_PSNR(mse):
    mselen=len(mse)
    psnr=[]
    for i in range(0,mselen):
        a=10*math.log((255**2)/mse[i],10)
        psnr.append(a)
    return psnr
b=Get_PSNR(a)
print(b)

def Get_MSE_perframe(video1,video2):
    videolen=len(video1)
    mse=[]
    for i in range(0,videolen):
        a=mean_squared_error(video1[i],video2[i])
        mse.append(a)

    return mse

def Get_q4():
    video=Get_Video(address='0',num_frame=150)
    video=To_Binary(video)
    video5=Get_Video(address='5',num_frame=150)
    video5 = To_Binary(video5)
    video10=Get_Video(address='10',num_frame=150)
    video10 = To_Binary(video10)
    video15=Get_Video(address='15',num_frame=150)
    video15 = To_Binary(video15)
    video20=Get_Video(address='20',num_frame=150)
    video20 = To_Binary(video20)
    video25=Get_Video(address='25',num_frame=150)
    video25 = To_Binary(video25)
    h5 = Get_MSE_perframe(video, video5)
    h10 = Get_MSE_perframe(video, video10)
    h15 = Get_MSE_perframe(video, video15)
    h20 = Get_MSE_perframe(video, video20)
    h25 = Get_MSE_perframe(video, video25)
    return h5,h10,h15,h20,h25

a1,b1,c1,d1,e1=Get_q4()
print('qd=5,mse=:')
print(a1)
print(len(a1))
print('qd=10,mse=:')
print(b1)
print(len(b1))
print('qd=15,mse=:')
print(c1)
print(len(c1))
print('qd=20,mse=:')
print(d1)
print(len(d1))
print('qd=25,mse=:')
print(e1)
print(len(e1))


biterate=[10901.63,5355.12,3784.63,2718.93,2211.47,1763.83,1522.87,1283.76,1147.70,
          1000.27,913.09,817.12,758.08,689.69,648.08,599.02,567.63,529.91,506.87,477.66,
          459.59,436.98,422.14,404.62,392.17,377.48,368.11,355.22,347.15,337.59]

plt.plot(biterate,a,c='r')
plt.scatter(biterate,a,c='k',s=5)
plt.xlabel('Bite Rate')
plt.ylabel('MSE')
plt.savefig('1.svg',type='svg')
plt.show()
plt.plot(biterate,b,c='r')
plt.xlabel('Bite Rate')
plt.ylabel('PSNR')
plt.scatter(biterate,b,c='k',s=5)
plt.savefig('2.svg',type='svg')
plt.show()






frameno=[]
for i in range(0,150):
    frameno.append(i)
plt.plot(frameno,a1,label='Bite Rate=2211.47 kb')
plt.scatter(frameno,a1,s=5)
plt.plot(frameno,b1,label='Bite Rate=1000.27 kb')
plt.scatter(frameno,b1,s=5)
plt.plot(frameno,c1,label='Bite Rate=648.08 kb')
plt.scatter(frameno,c1,s=5)
plt.plot(frameno,d1,label='Bite Rate=477.66 kb')
plt.scatter(frameno,d1,s=5)
plt.plot(frameno,e1,label='Bite Rate=392.17 kb')
plt.scatter(frameno,e1,s=5)
plt.legend()
plt.xlabel('Frame No.')
plt.ylabel('MSE')
plt.savefig('3.svg',type='svg')
plt.show()

