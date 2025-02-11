import os
import cv2
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

img = cv2.imread("0020.png")
img = cv2.resize(img,(512,512))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis(False)
plt.show()

sigma = 25
mean = 0
x,y,z = img_rgb.shape

gauss = np.random.normal(mean,sigma,(x,y,z)).astype(np.uint8)
noisy = cv2.add(img_rgb, gauss)

plt.imshow(noisy)
plt.axis(False)
plt.show()

def Gen_Gauss_Noise(image,noise_lvl = 25):
    x,y,z = image.shape
    mean = 0
    
    sigma = noise_lvl
    gauss = np.random.normal(mean,sigma,(x,y,z)).astype(np.uint8)
    noisy_img = cv2.add(image,gauss)
    
    return noisy_img

img = cv2.imread("0020.png")
img = cv2.resize(img,(512,512))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean_blur = cv2.blur(gray,(3,3))
median_blur = cv2.medianBlur(gray,ksize=3)
laplacian = cv2.Laplacian(gray,cv2.CV_64F)
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

plt.imshow(gray,cmap = "gray")
plt.title("Grayscale")
plt.show()
plt.imshow(mean_blur,cmap = "gray")
plt.title("Mean Blur")
plt.show()
plt.imshow(median_blur,cmap = "gray")
plt.title("Median Blur")
plt.show()
plt.imshow(laplacian,cmap = "gray")
plt.title("Laplacian")
plt.show()
plt.imshow(sobelx,cmap = "gray")
plt.title("Sobel X-axis")
plt.show()
plt.imshow(sobely,cmap = "gray")
plt.title("Sobel Y-axis")
plt.show()

def Feature_Collection(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mean_blur = cv2.blur(gray,(3,3))
    median_blur = cv2.medianBlur(gray,ksize=3)
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    
    feature_stack = np.stack([gray,mean_blur,median_blur,laplacian,sobelx,sobely])
    return feature_stack.reshape(-1, 6)

folder = "Data/DIV2K_train_HR"
X = []
y = []
for i in os.listdir(folder):
    img_path = os.path.join(folder,i)
    if img_path is None:
        continue
    
    img = cv2.imread(img_path)
    img = cv2.resize(img,(512,512))
    clean_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    noisy_img = Gen_Gauss_Noise(img)
    print(f"[{i}] - noisy image generated")
    
    feature_stack = Feature_Collection(noisy_img)
    print(f"[{i}] - feature stack created")
    
    X.append(feature_stack)
    y.append(clean_gray.flatten())
    print(f"[{i}] - Data appended!")
    
X_stack = np.vstack(X)
y_stack = np.hstack(y)

X_train,X_test,y_train,y_test = train_test_split(X_stack,y_stack,test_size=0.2,random_state=5)


model = xgb.XGBRegressor(objective="reg:squarederror",device = "gpu",n_estimators = 100,verbose = True)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")