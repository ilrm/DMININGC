import numpy as np
import matplotlib
import matplotlib.pyplot as plt


fLength = []
fWidth = []
fSize = []
fConc = []
fConc1 = []
fAsym = []
fM3Long = []
fM3Trans = []
fAlpha = []
fDist = []
num=0
filename = 'magic04.txt'
with open(filename) as file_to_read:
    while (True):
        lines = file_to_read.readline()
        num += 1
        if not lines:
            break
        fLt, fWt, fst, fct, fc1t, fat, fmlt, fmtt, fat, fdt,\
        attr = [i for i in lines.split(",")]
        fLength.append(float(fLt))
        fWidth.append(float(fWt))
        fSize.append(float(fst))
        fConc.append(float(fct))
        fConc1.append(float(fc1t))
        fAsym.append(float(fat))
        fM3Long.append(float(fmlt))
        fM3Trans.append(float(fmtt))
        fAlpha.append(float(fat))
        fDist.append(float(fdt))


y = [fLength,fWidth,fSize,fConc,fConc1,
     fAsym,fM3Long,fM3Trans,fAlpha,fDist]
k=np.array(y, dtype = float)
r=np.ndarray.dot(k,k.T)
print(np.cov(r))


"""
k=np.array(y, dtype = float)
r=np.outer(k,k)
print(np.cov(r))
"""

print(np.cov(y))
a=np.array(fLength)
b=np.array(fWidth)
plt.plot(a,b,'.')
plt.show()


data = a
mean = data.mean()
std = data.std()
def profun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf
x = np.arange(-200,200,0.1)
y = profun(x, mean, std)
plt.plot(x,y)
plt.hist(data, bins=10, rwidth=10, density=True)
plt.title('lab1')
plt.xlabel('Point')
plt.ylabel('Probability')               
plt.show()


def varifun(a):
    array = np.array(a)
    var = array.var()
    return var
var1=varifun(fLength)
var2=varifun(fWidth)
var3=varifun(fSize)
var4=varifun(fConc)
var5=varifun(fConc1)
var6=varifun(fAsym)
var7=varifun(fM3Long)
var8=varifun(fM3Trans)
var9=varifun(fAlpha)
var10=varifun(fDist)
c1=[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]
print (c1.index(max(c1)))
print (c1[c1.index(max(c1))])
print (c1.index(min(c1)))
print (c1[c1.index(min(c1))])



def covfun(a):
    array = np.array(a)
    cov = np.cov(array)
    return cov
cov1=covfun(fLength)
cov2=covfun(fWidth)
cov3=covfun(fSize)
cov4=covfun(fConc)
cov5=covfun(fConc1)
cov6=covfun(fAsym)
cov7=covfun(fM3Long)
cov8=covfun(fM3Trans)
cov9=covfun(fAlpha)
cov10=covfun(fDist)
c2=[cov1,cov2,cov3,cov4,cov5,cov6,cov7,cov8,cov9,cov10]
print (c2.index(max(c2)))
print (c2[c2.index(max(c2))])
print (c2.index(min(c2)))
print (c2[c2.index(min(c2))])
