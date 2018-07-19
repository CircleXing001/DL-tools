from PIL import Image
import os

def compressImage(srcPath,dstPath):  
    for filename in os.listdir(srcPath):  
        if not os.path.exists(dstPath):
                os.makedirs(dstPath)        
        srcFile=os.path.join(srcPath,filename)
        dstFile=os.path.join(dstPath,filename)
        if os.path.isfile(srcFile):    
            sImg=Image.open(srcFile) 
            w,h=sImg.size  
            dImg=sImg.resize((50,50),Image.ANTIALIAS)  
            dImg.save(dstFile) 
            print(dstFile+" compressed succeeded")
        if os.path.isdir(srcFile):
            compressImage(srcFile,dstFile)
if __name__=='__main__':  
    compressImage("nyu_depth_images","the_new")
