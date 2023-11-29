import cv2
import matplotlib.pylab as plt
import matplotlib.cm as cm
from model import StyleTransfer
"""
"gogh 1~4"
"kimhongdo 1"
"oil_paint 1~2"
"cartoon 1" 
"custom"

"k_means"
"in"
"black_and_white"
"bit"
"""
filterName = "gogh"
contentPath= "./data/content/con11.jpg"
content_img = cv2.imread(contentPath)
content_img = cv2.cvtColor(content_img,cv2.COLOR_BGR2RGB)
filterType = 1
fmodel = StyleTransfer(filter = filterName, 
                        filterType = filterType,
                        non_filtering_area = [[280,200],[500,300]]) # 0~500 y,x
res = fmodel.forward(content_img)

if filterName == "black_and_white":
    plt.imshow(res,cmap = cm.gray)
    plt.show()
else:
    plt.imshow(res)
    plt.show()