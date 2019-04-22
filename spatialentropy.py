import arcpy
import numpy as np
import math
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from scipy.stats import entropy as shannon_entropy

############################################################
#Reading raster image
def readtensor_1d(filename):
    raster=arcpy.Raster(filename)
    terrain = arcpy.RasterToNumPyArray(raster)
    terrain=terrain.reshape([terrain.size])
    if terrain.min() < 1:
        terrain-=terrain.min()
        terrain+=1
    return terrain
############################################################

############################################################
#Reading raster image
def readtensor_2d(filename):
    raster=arcpy.Raster(filename)
    terrain = arcpy.RasterToNumPyArray(raster)
    if terrain.min()<1:
        terrain-=terrain.min()
        terrain+=1
    return terrain
############################################################

############################################################
#Calculating Stirling
def calStirling(x):
    return x*math.log(x)-x+0.5*math.log(2*x*math.pi)
############################################################

############################################################
#斯特林公式计算近似熵
def calEntropy(tensor):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    
    tp=calStirling(tensor.size)

    bt=0
    for i in range(c.size):
        bt+=calStirling(c[i])
    return tp-bt

'''
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
calEntropy(dem)
'''
############################################################

############################################################
#energy_distance
def calemd_enegery(tensor):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:c[i]] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(tensor.size, np.float32)
    w = np.log(np.arange(tensor.size) + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] = h[i]
    #h1 = h1 * w #土堆总量
    
    #h2 = np.ones(h1.size) #土堆分布h1
    
    h2 = np.zeros(h1.size)
    h2[0] = h1.sum()
    
    distance=wasserstein_distance(h1, h2, w, w)
    ##配置熵的计算
    #N=np.log(np.arange(tensor.size)+1)
    #E=N.sum()-h1.sum()

    distance=energy_distance(h1, h2, w, w)
    dist_norm = 1 - distance/math.sqrt(2)
    return distance, dist_norm
'''
dem = readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
dist = calemd_enegery(dem)
print(dist)

dem = np.arange(800*800, dtype=np.int32)
dist = calemd_enegery(dem)
print(dist)

dem = np.zeros(800*800, dtype=np.int32) #对应最高熵
dist = calemd_enegery(dem)
print(dist)
'''
############################################################

############################################################
#wasserstein_distance
def calemd_wdist(tensor):
    size = tensor.size
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:c[i]] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(size, np.float32)
    w = np.log(np.arange(size) + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] = h[i]
    #h1 = h1 * w #土堆总量
    
    #h2 = np.ones(h1.size) #土堆分布h1
    
    h2 = np.zeros(h1.size)
    h2[0] = h1.sum()
    
    distance=wasserstein_distance(h1, h2, w, w)
    ##配置熵的计算
    #N=np.log(np.arange(tensor.size)+1)
    #E=N.sum()-h1.sum()
    dist_norm = 1 - distance

    return dist_norm
'''
dem = readtensor_1d(r'A:\dem\JDM_2.tif')

dist = calemd_wdist(dem)
print(dist)

dem = np.arange(800*800, dtype=np.int32)
dist = calemd_wdist(dem)
print(dist)

dem = np.zeros(800*800, dtype=np.int32) #对应最高熵
dist = calemd_wdist(dem)
print(dist)
'''
############################################################

############################################################
#wasserstein_distance [error]
def calemd_wdist_v2(tensor):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, c[i]-1] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(types, np.float32)
    _ = np.log(np.arange(types) + 1) #土堆密度
    w = np.arange(types) + 1 #土堆密度
    w = np.asarray(w, np.float32)
    for i in range(_.size):
        w[i] = np.sum(_[0:i+1])
    
    for i in range(h.size):
        h1[i]=h[i]
    #h1 = h1 * w #土堆总量
    
    #h2 = np.ones(h1.size) #土堆分布h1
    
    h2 = np.zeros(h1.size)
    h2[0]= h1.sum()
    
    distance=wasserstein_distance(h1, h2, w, w)
    ##配置熵的计算
    #N=np.log(np.arange(tensor.size)+1)
    #E=N.sum()-h1.sum()
    dist_norm = 1 - distance

    return distance, dist_norm
'''
dem = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
dist = calemd_wdist_v2(dem)
print(dist)

dem = np.arange(800*800, dtype=np.int32)
dist = calemd_wdist_v2(dem)
print(dist)

dem = np.zeros(800*800, dtype=np.int32) #对应最高熵
dist = calemd_wdist_v2(dem)
print(dist)
'''
############################################################

############################################################
def calemd_wdist_max(tensor_size):
    h1 = np.ones(tensor_size, np.float32)
    w = np.log(np.arange(tensor_size) + 1) #土堆密度
    #h1 = h1 * w #土堆总量
    
    h2 = np.zeros(h1.size)
    h2[0]= h1.sum()
    distance=wasserstein_distance(h1, h2, w, w)
    return distance
'''
dem = readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
dist = calemd_wdist(dem)
print(dist)

dem = np.arange(800*800, dtype=np.int32)
dist = calemd_wdist(dem)
print(dist)

dem = np.zeros(800*800, dtype=np.int32) #对应最高熵
dist = calemd_wdist(dem)
print(dist)

dem = readtensor_2d(r'A:\dem\China_DEM_Clip_west3.tif')
testDemEntropy(dem)

testEntropy(800, 800)
'''
#演示在全局熵一致的情况下，在局部熵
############################################################
def testDemEntropy(grid):
    
    dist_list = []
    dist_list = calMultiWdist(grid)
    
    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)
    
    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)
    
    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)

    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)
    
    return dist_list
############################################################


############################################################

def testEntropy(rows, cols):
    dem = np.zeros([rows, cols], dtype=np.int32)
    grid = split2d(dem, 2, 2)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j]=i*2+j
    
    grid=grid.swapaxes(1,2)
    grid=grid.reshape(rows, cols)
    
    dist_list = []
    dist_list = calMultiWdist(grid)
    
    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)
    
    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)
    
    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)

    np.random.shuffle(grid)
    dist_list = calMultiWdist(grid)
    
    return dist_list
############################################################


############################################################
def split2d(A, rows, cols):
    grid=[]
    lines =[row for row in np.split(A, rows, axis=0)]
    for part in lines:
        grid.append([part for part in np.split(part, cols, axis=1)])
    grid = np.array(grid)   
    return grid
############################################################

############################################################
def pooling2d(A, rows, cols):
    grid=[]
    lines =[row for row in np.split(A, rows, axis=0)]
    for part in lines:
        grid.append([part for part in np.split(part, cols, axis=1)])
    grid = np.array(grid)
    for i in range(rows):
        for j in range(cols):
            '''
            hist = np.histogram(grid[i, j])
            x = np.where(hist[0].max())
            grid[i, j] = np.take(hist[1], x)
            '''
            grid[i, j] = grid[i, j].mean()
    
    grid = grid.reshape(rows, -1)
    grid = grid.reshape(grid.size)
    return grid
    
'''
    #dem = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
    #grid = pooling2d(dem, 200, 200)
    #print(calemd_wdist(grid))
    
    pooling_list = []
    dem = readtensor_2d(r'A:\dem\China_DEM_Clip_east1.tif')
    dist_list = calMultiWdist(dem)
    pooling_list.append(dist_list)
    
    dem = readtensor_2d(r'A:\dem\China_DEM_Clip_east2.tif')
    dist_list = calMultiWdist(dem)
    pooling_list.append(dist_list)
    
    dem = readtensor_2d(r'A:\dem\China_DEM_Clip_east3.tif')
    dist_list = calMultiWdist(dem)
    pooling_list.append(dist_list)
    
    dem = readtensor_2d(r'A:\dem\China_DEM_Clip_west1.tif')
    dist_list = calMultiWdist(dem)
    pooling_list.append(dist_list)
    
    dem = readtensor_2d(r'A:\dem\China_DEM_Clip_west2.tif')
    dist_list = calMultiWdist(dem)
    pooling_list.append(dist_list)
    
    dem = readtensor_2d(r'A:\dem\China_DEM_Clip_west3.tif')
    dist_list = calMultiWdist(dem)
    pooling_list.append(dist_list)

    #draw_pooling_tends(pooling_list)
'''
############################################################

############################################################
def calMultiWdist(dem):
    dist_list = []
    
    grid = pooling2d(dem, 1, 1)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)

    grid = pooling2d(dem, 2, 2)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)

    grid = pooling2d(dem, 4, 4)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)
    
    grid = pooling2d(dem, 8, 8)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)

    grid = pooling2d(dem, 16, 16)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)

    grid = pooling2d(dem, 32, 32)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)
    
    grid = pooling2d(dem, 40, 40)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)
    
    grid = pooling2d(dem, 80, 80)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)
    
    grid = pooling2d(dem, 160, 160)
    dist = calemd_wdist(grid)
    pooling_list.append(dist)
    print(dist)
    
    grid = pooling2d(dem, 200, 200)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)
    
    grid = pooling2d(dem, 400, 400)
    dist = calemd_wdist(grid)
    dist_list.append(dist)
    print(dist)
    
    grid = pooling2d(dem, 800, 800)
    print(calemd_wdist(grid))
    dist_list.append(dist)
    return pooling_list
    
    print()
############################################################

############################################################
def draw_pooling_tends(pooling_list):
    plt.xlabel('pooling cell size')
    plt.ylabel('wasserstein distance')
    locs=np.arange(12)
    labels=('1', '2', '4', '8', '16', '32', '40', '80', '160', '200', '400', '800')
    plt.xticks(locs, labels)
    for plist in pooling_list:
        plt.plot(plist, linewidth=1, color='b', label='Frequency')
    plt.show()
'''
0.00 	0.00 	0.00 	0.00 	0.00 	0.00 
3.00 	3.00 	1.00 	3.00 	3.00 	3.00 
4.33 	7.00 	7.00 	15.00 	15.00 	15.00 
15.00 	11.80 	8.14 	31.00 	31.00 	20.33 
27.44 	18.69 	7.83 	84.33 	84.33 	41.67 
43.52 	16.66 	7.39 	145.29 	203.80 	59.24 
47.48 	16.20 	7.65 	159.00 	265.67 	75.19 
76.11 	14.38 	7.39 	219.69 	580.82 	100.59 
95.24 	14.01 	7.28 	300.18 	751.94 	114.84 
101.04 	13.07 	7.18 	357.74 	739.74 	113.29 
107.70 	12.06 	6.82 	357.74 	736.33 	107.62 
113.29 	11.75 	6.91 	371.74 	736.33 	105.60 

'''
############################################################

#############################################################
#图像边界处理
def fiximage(filename1, filename2):    
    raster=arcpy.Raster(filename1)
    terrain = arcpy.RasterToNumPyArray(raster)

    if terrain.shape[0]>400:
        terrain[:, 799]=terrain[:, 798]
    corner=arcpy.Point(raster.extent.XMin, raster.extent.YMin)
    
    fixed_image=arcpy.NumPyArrayToRaster(terrain, corner, \
      raster.meanCellWidth, raster.meanCellHeight, raster.noDataValue)
    fixed_image.save(filename2)
############################################################

#############################################################
#图像转要素
def img2polygon(filename):
    raster=arcpy.Raster(filename)
    arcpy.conversion.RasterToPolygon(raster, simplify=False, raster_field='value')
#############################################################

#############################################################
#要素转向量
def ply2img(fc):
    fields = ['Id', 'gridcode', 'Shape_Area']
    arr = arcpy.da.FeatureClassToNumPyArray(fc, fields)
    #count = arr['count']
    tid = arr['gridcode']
    area = arr['Shape_Area']
    unit =arr['Shape_Area'].min()
    if tid.size > 100:
        cid = area/unit
    else:
        cid = arr['Shape_Area']
    cid = np.around(cid)
    cid = np.array(cid, np.int32) #不能直接转会出现偏差
    return tid, cid

'''
fc = r'A:\test.gdb\RasterT_China_D1'
tid, cid = ply2img(fc)
'''
############################################################

############################################################
#计算要素分布的wasserstein_distance
def calDistance1(tid, cid):
    cid_types=np.unique(cid) #斑块的分布
    tid_types=np.unique(tid) #要素的分布
    
    #计算要素的histogram
    f=np.zeros(tid_types.size, dtype=np.int32)
    for (tid_id, i) in zip(tid_types, range(tid_types.size)):
        pos=np.where(tid==tid_id) #定位斑块
        cid_focused=cid[pos[0]] #每个斑块上的要素数目
        
        #cid_focused=np.ones(cid_focused.shape) #强制取1
        f[i]=cid_focused.sum() #计算要素的histogram

    #要素类型排列的分解
    e=np.zeros([tid_types.size, f.max()], dtype=np.int32)
    for i in range(tid_types.size):
        e[i, 0:f[i]] = 1
        
    # 组合分布e.sum(0)
    # 要素分布e.sum(1) 

    h1 = e.sum(0)
    h2 = np.ones(h1.shape, dtype=np.float32)
    distance=wasserstein_distance(h1, h2)
    return distance
'''
fc = r'A:\test.gdb\RasterT_China_D1'
tid, cid = ply2img(fc)
calDistance1(tid, cid)
'''
############################################################

############################################################
#计算要素分布的wasserstein_distance [剔除空间法]
def calDistance2(tid, cid):
    cid_types=np.unique(cid) #斑块的分布
    tid_types=np.unique(tid) #要素的分布
    
    #计算要素的histogram
    f=np.zeros(tid_types.size, dtype=np.int32)
    for (tid_id, i) in zip(tid_types, range(tid_types.size)):
        pos=np.where(tid==tid_id) #定位斑块
        cid_focused=cid[pos[0]] #每个斑块上的要素数目
        
        #cid_focused=np.ones(cid_focused.shape) #强制取1
        f[i]=cid_focused.sum() #计算要素的histogram

    #要素类型排列的分解
    e=np.zeros([tid_types.size, f.max()], dtype=np.int32)
    for i in range(tid_types.size):
        e[i, 0:f[i]] = 1
        
    # 组合分布e.sum(0)
    # 要素分布e.sum(1) 

    h1 = e.sum(0)
    h2 = np.ones(h1.shape, dtype=np.float32)
    distance=wasserstein_distance(h1, h2)
    return distance
'''
csv_name = r'A:\csv\spatial_5.txt'
fc = r'A:\test.gdb\spatial_5'
csv2ply(csv_name, fc)

fc = r'A:\test.gdb\RasterT_China_D6'

tid, cid = ply2img(fc)
calDistance2(tid, cid)
'''
############################################################

############################################################
#计算要素分布的wasserstein_distance #计算空间的熵和要素的熵
def calDistance3(tid, cid):
    #场景的大小
    size = cid.sum()
    
    #扣除斑块的重复
    types=cid.size
    h=np.zeros([cid.max()], dtype=np.int32)
    for i in range(types):
        h[0:cid[i]] += 1

    h1 = np.zeros(size, np.float32)
    w = np.log(np.arange(size) + 1) #土堆密度
    if w[0]==0:
        w[0] = 1e-9
    
    for i in range(h.size):
        h1[i] = h[i]
    
    h2 = np.zeros(h1.size)
    
    dist_spatial = 1 - wasserstein_distance(h1, h2, w, w)
    

    #扣除类型的重复
    c = np.bincount(tid, cid)
    c = np.around(c)
    c = np.array(c, np.int32) #不能直接转会出现偏差
    c = c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    h=np.zeros([c.max()], dtype=np.int32)
    for i in range(types):
        h[0:c[i]] += 1
    h1 = np.zeros(size, np.float32)
    
    for i in range(h.size):
        h1[i] = h[i]

    h2 = np.zeros(h1.size)
    dist_feature = 1 - wasserstein_distance(h1, h2, w, w)
    
    return dist_spatial, dist_feature, dist_spatial*dist_feature
'''
csv_name = r'A:\csv\spatial_3.txt'
fc = r'A:\test.gdb\li2002_sim_2'

#fc = r'A:\test.gdb\RasterT_China_D1'
#fc = r'A:\test.gdb\spatial_1'
tid, cid = ply2img(fc)
tid -= tid.min()
tid += 1
calDistance3(tid, cid)
'''
############################################################

############################################################
def csv2ply(filename, fc):
    data = np.loadtxt(filename, delimiter=',', dtype='int')
    corner = arcpy.Point(0, 0)
    ras = arcpy.NumPyArrayToRaster(data, corner, x_cell_size=1, y_cell_size=1, value_to_nodata=-9999)
    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)
    arcpy.conversion.RasterToPolygon(ras, fc, 'NO_SIMPLIFY', 'VALUE')
'''
csv_name = r'A:\csv\spatial_4.txt'
fc = r'A:\test.gdb\spatial_4'
csv2ply(csv_name, fc)
tid, cid = ply2img(fc)
calDistance3(tid, cid)
'''
############################################################

############################################################calCVEMD
def calCVEMD(h):
#w1、w2是两个矩阵，第一列表示权值，后面三列表示直方图或数量
    w1=np.zeros([1, 100000], dtype=np.float32)
    w2=np.ones([1, 100000], dtype=np.float32)

    w1[0,0]=1
    for i in range(h.shape[0]):
        w1[0,i+1]=h[i]

    emd=cv.EMD(w1,w2,cv.DIST_L2)
    return emd
############################################################

############################################################
#计算要素分布的wasserstein_distance #联合熵
def calDistance4(tif, tid, cid):
    #场景的大小
    size = cid.sum()
    
    #扣除类型的重复
    c=np.bincount(tif)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:c[i]] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(size, np.float32)
    w = np.log(np.arange(size) + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] = h[i]
    #h1 = h1 * w #土堆总量
    #h2 = np.ones(h1.size) #土堆分布h1

    #扣除斑块的重复
    #c=np.bincount(cid)
    #c=cid[cid.nonzero()] #得到类型的histogram
    types=cid.size
    
    #
    grid=np.zeros([types, cid.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:cid[i]] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    #h1 = np.zeros(size, np.float32)
    w = np.log(np.arange(size) + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] += h[i]
    #h1 = h1 * w #土堆总量
    #h2 = np.ones(h1.size) #土堆分布h1
    
    h2 = np.zeros(h1.size)
    
    distance = wasserstein_distance(h1, h2, w, w)
    dist_norm = 1 - distance
    return distance, dist_norm
'''
tif = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
fc = r'A:\test.gdb\RasterT_China_D6'
tid, cid = ply2img(fc)
dist = calDistance4(tif, tid, cid)
'''
############################################################

############################################################
#计算要素分布的wasserstein_distance [空间挤压法]
def calDistance5(tid, cid):
    cid_types=np.unique(cid) #斑块的分布
    tid_types=np.unique(tid) #要素的分布
    
    #计算要素的histogram
    f=np.zeros(tid_types.size, dtype=np.int32)
    for (tid_id, i) in zip(tid_types, range(tid_types.size)):
        pos=np.where(tid==tid_id) #定位斑块
        cid_focused=cid[pos[0]] #每个斑块上的要素数目
        
        cid_focused=np.ones(cid_focused.shape) #强制取1
        f[i]=cid_focused.sum() #计算要素的histogram

    #要素类型排列的分解
    e=np.zeros([tid_types.size, f.max()], dtype=np.int32)
    for i in range(tid_types.size):
        e[i, 0:f[i]] = 1
        
    # 组合分布e.sum(0)
    # 要素分布e.sum(1) 

    h1 = e.sum(0)
    h2 = np.ones(h1.shape, dtype=np.float32)
    distance=wasserstein_distance(h1, h2)*cid.size/cid.sum()
    return distance
'''
fc = r'A:\test.gdb\RasterT_China_D1'
tid, cid = ply2img(fc)
calDistance5(tid, cid)
'''
############################################################

############################################################
#计算要素分布的wasserstein_distance #联合熵
def calDistance6(tid, cid):
    #场景的大小
    size = cid.sum()
    
    #扣除类型的重复
    c=np.bincount(tid, cid)
    c = np.around(c)
    c = np.array(c, np.int32) #不能直接转会出现偏差
    c = c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:c[i]] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(size, np.float32)
    w = np.log(np.arange(size) + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] = h[i]
    #h1 = h1 * w #土堆总量
    #h2 = np.ones(h1.size) #土堆分布h1

    #扣除斑块的重复
    #c=np.bincount(cid)
    #c=cid[cid.nonzero()] #得到类型的histogram
    types=cid.size
    
    #
    grid=np.zeros([types, cid.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:cid[i]] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    #h1 = np.zeros(size, np.float32)
    w = np.log(np.arange(size) + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] += h[i]
    #h1 = h1 * w #土堆总量
    #h2 = np.ones(h1.size) #土堆分布h1
    
    h2 = np.zeros(h1.size)
    h2[0] = h1.sum()
    h1 = h1/2
    h2 = h2/2
    distance = wasserstein_distance(h1, h2, w, w)
    
    dist_norm = 1 - distance
    return distance, dist_norm
'''
fields = ['Id', 'gridcode', 'Shape_Area']
#fc = r'A:\test.gdb\spatial_3'
fc = r'A:\test.gdb\li2002_sim_1'
arr = arcpy.da.FeatureClassToNumPyArray(fc, fields)

tid = arr['gridcode']
tid-= (tid.min()-1)

cid = (arr['Shape_Area']/arr['Shape_Area'].min())
#cid = (arr['Shape_Area'])
cid = np.around(cid)
cid = np.array(cid, np.int32) #不能直接转会出现偏差

dist = calDistance6(tid, cid)
'''
############################################################


############################################################
def calF(tensor, idx=0):
    types=tensor.max()
    c=np.histogram(tensor, types+1)
    t=np.nonzero(c[0])
    e=np.zeros([types+1, c[0].max()+1], dtype=np.int32)
    for i in range(types):
            e[i, 0:c[0][i]] = 1
    if idx==0:
        f = np.sum(e, 0)
    else:
        f = np.sum(e, 1)
    return f
############################################################

############################################################

def draw_complexity_subplot_histogram(list, dir, filename):
    plt.close()
    fig, subplts = plt.subplots(nrows=3, ncols=2)
    subplts = subplts.reshape(subplts.size)
    for i in range(subplts.size):
        subplts[i].set_xlabel('Logarithm Intensity')
        subplts[i].set_ylabel('Counts')
        subplts[i].plot(calF(list[i], 0), linewidth=1, color='b', label='Frequency')
    fig.subplots_adjust(hspace=0.8)
    fig.subplots_adjust(wspace=0.5)
    fig.align_labels()
    
    #fig.tight_layout
    
    plt.savefig(dir+filename, dpi=300)
'''
list=[]
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east2.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west1.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west2.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
list.append(dem)

dir = r'A:\graph\\'
filename = r'complexity.png'
draw_complexity_subplot_histogram(list, dir, filename)
'''

############################################################


############################################################
def draw_type_subplot_histogram(list, dir, filename):
    plt.close()
    fig, subplts = plt.subplots(nrows=3, ncols=2)
    subplts = subplts.reshape(subplts.size)
    for i in range(subplts.size):
        subplts[i].set_xlabel('Landscape Types')
        subplts[i].set_ylabel('Counts')
        subplts[i].plot(calF(list[i], 1), linewidth=1, color='b', label='Frequency')
    fig.subplots_adjust(hspace=0.8)
    fig.subplots_adjust(wspace=0.5)
    fig.align_labels()
    
    #fig.tight_layout
    
    plt.savefig(dir+filename, dpi=300)
'''
list=[]
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east2.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west1.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west2.tif')
list.append(dem)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
list.append(dem)

dir = r'A:\graph\\'
filename = r'type.png'
draw_type_subplot_histogram(list, dir, filename)
'''

############################################################


############################################################
def draw_type_histogram(tensor, dir, filename):
    plt.close()
    plt.xlabel('Type')
    plt.ylabel('Numbers')
    plt.plot(calF(tensor, 1), linewidth=1, color='b', label='Frequency')
    plt.savefig(dir+filename, dpi=300)
    #plt.show()
'''
##绘制类型分布图
dir = r'A:\graph\\'
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
filename = r'type_1.png'
draw_type_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east2.tif')
filename = r'type_2.png'
draw_type_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
filename = r'type_3.png'
draw_type_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west1.tif')
filename = r'type_4.png'
draw_type_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west2.tif')
filename = r'type_5.png'
draw_type_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
filename = r'type_6.png'
draw_type_histogram(dem, dir, filename)

'''


'''
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
a=calF(dem, 0)
a.tofile(r'a:\log\dem_1.csv', ',')
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east2.tif')
a=calF(dem, 0)
a.tofile(r'a:\log\dem_2.csv', ',')
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
a=calF(dem, 0)
a.tofile(r'a:\log\dem_3.csv', ',')
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west1.tif')
a=calF(dem, 0)
a.tofile(r'a:\log\dem_4.csv', ',')
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west2.tif')
a=calF(dem, 0)
a.tofile(r'a:\log\dem_5.csv', ',')
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
a=calF(dem, 0)
a.tofile(r'a:\log\dem_6.csv', ',')
'''
############################################################

############################################################
def draw_complexcity_histogram(tensor, dir, filename):
    plt.close()
    plt.xlabel('Logarithmic complexity')
    plt.ylabel('Configure repeats')
    plt.plot(calF(tensor, 0), linewidth=1, color='b', label='Frequency')
    plt.savefig(dir+filename, dpi=300)
    #plt.show()
'''
##绘制复杂分布图
dir = r'A:\graph\\'
dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
filename = r'Logarithmic complexity D1.png'
draw_complexcity_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east2.tif')
filename = r'Logarithmic complexity D2.png'
draw_complexcity_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
filename = r'Logarithmic complexity D3.png'
draw_complexcity_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west1.tif')
filename = r'Logarithmic complexity D4.png'
draw_complexcity_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west2.tif')
filename = r'Logarithmic complexity D5.png'
draw_complexcity_histogram(dem, dir, filename)

dem=readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
filename = r'Logarithmic complexity D6.png'
draw_complexcity_histogram(dem, dir, filename)
'''
############################################################
def calH(tensor):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:c[i]] = 1

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(tensor.size, np.float32)
    w = np.log(np.arange(tensor.size) + 1) #土堆密度
    
    for i in range(h.size):
        h1[i]=h[i]
    h1 = h1 * w #土堆总量
    return h1
############################################################

############################################################
def compH(tensor1, tensor2):
    
    h1 = calH(tensor1)
    h2 = calH(tensor2)
    
    distance = wasserstein_distance(h1, h2)
    dist_norm = distance/calemd_wdist_max(tensor.size)

    return distance, dist_norm
    
'''
dem1 = readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
dem2 = readtensor_1d(r'A:\dem\China_DEM_Clip_east2.tif')
dem3 = readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
dem4 = readtensor_1d(r'A:\dem\China_DEM_Clip_west1.tif')
dem5 = readtensor_1d(r'A:\dem\China_DEM_Clip_west2.tif')
dem6 = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')

dist_list=[]
dem_list=[dem1, dem2, dem3, dem4, dem5, dem6]
dem_name=['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
for i in range(6):
    for j in range(6):
        dist, dist_norm = compH(dem_list[i], dem_list[j])
        dist_list.append([dem_name[i],dem_name[j], dist, dist_norm])
print(dist_list)
'''
############################################################

############################################################
def calShanon(tensor):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    s = shannon_entropy(c/c.size)
    e = math.log(tensor.size) - (c * np.log(c)).sum() /tensor.size
    return s, e
'''
tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
dist = calShanon(tensor)
print(dist)

tensor = np.arange(tensor.size, dtype=np.int32) ##熵最大
dist = calShanon(tensor)
print(dist)

tensor = np.zeros(tensor.size, dtype=np.int32) #熵为零
dist = calShanon(tensor)
print(dist)

'''
############################################################

############################################################
def calShanon_wdist(tensor):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size

    h1 = np.zeros(tensor.size, np.float32)
    w = np.log(np.arange(tensor.size) + 1) #土堆密度
    c.sort()
    
    for i in range(c.size):
        h1[i] = c[i]
        w[i] = np.log(c[i])
        w[0] = 1e-9
    
    #h2 = np.ones(h1.size)

    h2 = np.ones(h1.size)
    #h2[0]= h1.sum()

    distance=wasserstein_distance(h1, h2, w, w)
    return distance

'''
tensor = readtensor_1d(r'A:\dem\c5_11.tif')
dist = calShanon(tensor)
print(dist)

tensor = np.arange(tensor.size, dtype=np.int32) ##熵最大
dist = calShanon(tensor)
print(dist)

tensor = np.zeros(tensor.size, dtype=np.int32) #熵为零
dist = calShanon(tensor)
print(dist)

tensor = np.full(1000, 1)
dist = shannon_entropy(tensor)
print(dist)


tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
dist = calShanon_wdist(tensor)
print(dist)


tensor = np.arange(tensor.size, dtype=np.int32) ##熵最大
dist = calShanon_wdist(tensor)
print(dist)

tensor = np.zeros(tensor.size, dtype=np.int32) #熵为零
dist = calShanon_wdist(tensor)
print(dist)

tensor = np.random.randint(0, tensor.size, size=tensor.size).astype(np.int32) #熵为零
dist = calShanon_wdist(tensor)
print(dist)


'''
############################################################

############################################################
def calShanon_wdist2(tensor):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:c[i]] = c[i]

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(tensor.size, np.float32)
    w = np.arange(tensor.size).astype(np.float32)
    w = np.reciprocal(w + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] = h[i]
    
    E = h1 * w #土堆总量
    S = math.log(h1.size) - E.sum()/tensor.size + 0.577217169664145
    
    #h2 = np.ones(h1.size) * w #准备填充的土坑
    h2 = np.zeros(h1.size)
    h2[0]= h1.sum()
    
    distance = wasserstein_distance(h1, h2, w, w)/h1.size
    dist_norm = 1 - distance/(calShanon_wdist_max(h1.size)+1)
    return distance, dist_norm, S
'''
tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
dist = calShanon_wdist2(tensor)
print(dist)

tensor = np.arange(tensor.size, dtype=np.int32) ##熵最大
dist = calShanon_wdist2(tensor)
print(dist)

tensor = np.zeros(tensor.size, dtype=np.int32) #熵为零
dist = calShanon_wdist2(tensor)
print(dist)

tensor = np.random.randint(0, tensor.size, size=tensor.size).astype(np.int32) #熵最大
dist = calShanon_wdist2(tensor)
print(dist)

dist = calShanon(tensor)
print(dist)


tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
dist = calShanon_wdist2(tensor)
print(dist)


np.random.shuffle(tensor)
dist = calShanon_wdist2(tensor)
print(dist)
'''



'''
tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_east1.tif')
dist = calShanon_wdist2(tensor)
print(dist)

dist = calShanon(tensor)
print(dist)

tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_east2.tif')
dist = calShanon_wdist2(tensor)
print(dist)

dist = calShanon(tensor)
print(dist)

tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_east3.tif')
dist = calShanon_wdist2(tensor)
print(dist)

dist = calShanon(tensor)
print(dist)

tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_west1.tif')
dist = calShanon_wdist2(tensor)
print(dist)

dist = calShanon(tensor)
print(dist)

tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_west2.tif')
dist = calShanon_wdist2(tensor)
print(dist)

dist = calShanon(tensor)
print(dist)

tensor = readtensor_1d(r'A:\dem\China_DEM_Clip_west3.tif')
dist = calShanon_wdist2(tensor)
print(dist)

dist = calShanon(tensor)
print(dist)
'''
############################################################

############################################################
def calShanon_wdist_max(tensor_size):
    h1 = np.ones(tensor_size, np.float32)
    w = np.reciprocal(np.arange(tensor.size) + 1) #土堆密度
    #h1 = h1 * w #土堆总量
    
    #当种类最大并且每个数量为1
    h2 = np.zeros(h1.size)
    h2[0]= h1.sum()
    distance=wasserstein_distance(h1, h2, w, w)
    return distance
############################################################

############################################################
def calShanon_wdist_max2(tensor_size):
    c=np.bincount(tensor)
    c=c[c.nonzero()] #得到类型的histogram
    types=c.size
    
    #
    grid=np.zeros([types, c.max()], dtype=np.int32)
    for i in range(types):
        grid[i, 0:c[i]] =c[i]
    grid = grid / tensor.size

    h = np.sum(grid, 0).astype(np.float32) #土堆分布

    h1 = np.zeros(tensor.size, np.float32)
    w = np.arange(tensor.size).astype(np.float32)
    w = np.reciprocal(w + 1) #土堆密度
    
    for i in range(h.size):
        h1[i] = h[i]
    
    E = h1 * w #土堆总量
    S = math.log(h1.size) - E.sum() + 0.577217169664145
    
    #h2 = np.ones(h1.size) * w #准备填充的土坑
    h2 = np.zeros(h1.size)
    h2[0]= h1.sum()
    
    distance = wasserstein_distance(h1, h2, w, w)
    return distance
    
############################################################

############################################################
def calLogN(N):
    sum = 0
    for i in range(N):
        sum+=1/(i+1)
    sum-=0.577217169664145
    return sum, math.log(N)
'''
print(calLogN(10))
print(calLogN(20))
print(calLogN(30))
print(calLogN(40))
print(calLogN(50))
print(calLogN(60))
print(calLogN(70))
print(calLogN(80))
print(calLogN(1000))
print(calLogN(10000))
print(calLogN(100000))
print(calLogN(1000000))
'''

############################################################
############################################################
def calNLogN(N):
    sum = 0
    for i in range(N):
        sum+=math.log(i+1)
    if N > 50:
        sum+=N-3
    elif N >20:
        sum+=N-3
    elif N >10:
        sum+=N-3
    else:
        sum+=N-1
    return sum, N*math.log(N)
'''
for i in range(1001):
    print(calNLogN(i+1))
'''
############################################################