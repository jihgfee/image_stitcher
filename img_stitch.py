import glob
import numpy as np
import cv2
import os
import sys

## ----------------------------------------------------
## Functions
## ----------------------------------------------------

def paint_background(img,(b,g,r,a)):#fill_color):#(b,g,r,a)):
    
    h,w,d = img.shape
 
    # Create imgs
    foreground = img[:,:,:3]
    background = np.full((h,w,3), [b,g,r],dtype=np.uint8)
    alpha = img[:,:,3]

    # Prepare imgs
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Prepare Alpha
    alpha = alpha.astype(float)/255
    alpha = cv2.merge((alpha,alpha,alpha))

    # Do blending
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    foreground = cv2.add(foreground, background)
    
    # Add constant alpha channel
    b_channel, g_channel, r_channel = cv2.split(foreground)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    return img

def back_to_mean(img):
    b,g,r,a = cv2.mean(img)
    return paint_background(img, [b,g,r,255])

def save_pos(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        img = origImg.copy()
        cv2.circle(img,(x,y),5,(0,255,0),-1)
        ix,iy = x,y
        cv2.imshow("img",img)

def compute_scalar((imgH,imgW), (imgHeight,imgLength),(hMid,wMid)): 
    imgHMid = imgH / 2 if hMid == None else hMid
    imgWMid = imgW / 2 if wMid == None else wMid
    
    hNegDiff = imgHMid
    hPosDiff = imgH - imgHMid
    wNegDiff = imgWMid
    wPosDiff = imgW - imgWMid

    minWDiff = min(wNegDiff, wPosDiff)
    minHDiff = min(hNegDiff, hPosDiff)
    
    wScalar = (float(imgLength) / float(2)) / float(minWDiff)
    hScalar = (float(imgHeight) / float(2)) / float(minHDiff)
    maxScalar = max(wScalar, hScalar)
    return maxScalar
    
def add_image(newImg, (xPos,yPos), (imgHeight,imgWidth), img, (hMid,wMid)):
    imgH,imgW,imgD = img.shape
    
    imgHMid = int(imgH / 2) if hMid == None else hMid
    imgWMid = int(imgW / 2) if wMid == None else wMid

    imgHS = imgHMid - imgHeight/2
    imgHS = imgHS if imgHS >= 0 else 0 # Hack, pls fix
    imgHE = imgHS + imgHeight
    imgWS = imgWMid - imgWidth/2
    imgWS = imgWS if imgWS >= 0 else 0 # Hack, pls fix
    imgWE = imgWS + imgWidth

    newImg[yPos:yPos+imgHeight,xPos:xPos+imgWidth,:] = img[imgHS:imgHE,imgWS:imgWE,:]

## ----------------------------------------------------
## Setup
## ----------------------------------------------------

# Direction "Enum"
hDir = "horizontal"
vDir = "vertical"

# Initialize globals
ix,iy = -1,-1
fileTypes = ["png"]#,"jpg"]

# Get paramters from args
path = sys.argv[1]
w,h = tuple(int(i) for i in sys.argv[2][1:-1].split(','))
barLength = 2
direction = sys.argv[3]

# Change to directory
owd = os.getcwd()
os.chdir(path)

# Get all images in folder
imgs = []
for ft in fileTypes:
    imgs.extend(glob.glob("*."+ft))

# Initialize compute parameters
imgCount = len(imgs)
totalBarLength = (imgCount-1) * barLength
lengthNoBar = w - totalBarLength if direction == hDir else h - totalBarLength if direction == vDir else None
remainder = lengthNoBar % imgCount

# Length and height of each image
imgWidth = lengthNoBar / imgCount if direction == hDir else w
imgHeight = lengthNoBar / imgCount if direction == vDir else h

# Initialize new image
newImg = np.zeros((h,w,4), np.uint8)
xPos = 0
yPos = 0

config_prefix = "h" if direction == hDir else "v" if direction == vDir else None
config_name = config_prefix+"_config.txt"

# Create and read config
cfg_exists = os.path.isfile(config_name)
if not cfg_exists:
    cfg_file = open(config_name,"w+")
    cfg_file.close()

cfg_map = dict()
cfg_file = open(config_name,"r")
for line in cfg_file.readlines():
    f,xy = line.split(":")
    x,y = xy.split(",")
    cfg_map[f] = (int(x),int(y))
cfg_file.close()

for file in imgs:
    print "Processing: "+file
    
    # Read image
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED) 

    # Repaint background
    img = back_to_mean(img)

    # Consider remainders
    realHeight = imgHeight + 1 if remainder > 0 and direction == vDir else imgHeight
    realWidth = imgWidth + 1 if remainder > 0 and direction == hDir else imgWidth
    
    # Get middle from config or input
    if(file in cfg_map):
        ix,iy = cfg_map[file]
    else:
        origImg = img.copy()
        cv2.namedWindow("img")
        cv2.setMouseCallback("img", save_pos)
        cv2.imshow("img",img)
        cv2.waitKey(0) 
    
        line = file + ": " + str(ix) + "," + str(iy) + "\n"
        with open(config_name, "a") as cfg_file:
            cfg_file.write(line)

    # Scale image
    scalar = compute_scalar(img.shape[:2],(imgHeight,realWidth),(None if iy < 0 else iy, None if ix < 0 else ix))
    img = cv2.resize(img, (0,0), fx=scalar, fy=scalar)

    # Add image to new image
    add_image(newImg, (xPos,yPos), (realHeight,realWidth), img, (None if iy < 0 else int(iy*scalar), None if ix < 0 else int(ix * scalar)))
    
    # Update iterators
    remainder = remainder - 1
    ix,iy = -1,-1
    xPos = xPos + realWidth + barLength if direction == hDir else xPos
    yPos = yPos + realHeight + barLength if direction == vDir else yPos

cv2.imshow("img",newImg)

cv2.waitKey(0)

os.chdir(owd)
cv2.imwrite("img.png",newImg)
