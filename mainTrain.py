#Importing The Libraries
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
import pickle

# //Set Dataset Path according to yourself
image_path = './dataset'  

#Two Arrays for (Images and Labels->{0,1}) (0 = No Tumor  , 1 = Tumor Present)
dataset = []
label = []


#Importing data into arrays
no_tumor_list = os.listdir(image_path + '/no/')
yes_tumor_list = os.listdir(image_path + '/yes/')

for i,image_name in enumerate(no_tumor_list):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_path+'/no/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((60,60))
        dataset.append(np.array(image))
        label.append(0)

for i,image_name in enumerate(yes_tumor_list):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_path+'/yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((60,60))
        dataset.append(np.array(image))
        label.append(1)     

#Defining The Values        
dec = {0:'No Tumor', 1:'Tumor Detected'}

#Converting Array to NumPy array (to useNumPy modules)
dataset = np.array(dataset)
label = np.array(label)

#Our ML Algo. works only on 2-D dataset so converting it to 2-D dataset  
dataset = dataset.reshape(len(dataset),-1)

#Splitting the data to Train/Test split (Train:80%,Test:20%)
x_train,x_test,y_train,y_test = train_test_split (dataset,label,test_size=0.2,random_state=0)

#You will get the maximum value in (x_train,x_test,y_train,y_test) => 255 (as the max. value of RGB image is 255)
print(x_train.max(), x_train.min())
print(x_test.max(), x_test.min())

#Divide the values with max value to make it complete Binary(0,1)
x_train = x_train/255
x_test = x_test/255

#Algorithm Testing
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(x_train, y_train)

# sv = SVC()
# sv.fit(x_train, y_train)

# Training and Testing Score of LG
# print("Training Score:", lg.score(x_train, y_train))
# print("Testing Score:", lg.score(x_test, y_test))
# Training Score: 0.9920833333333333
# Testing Score: 0.97


# Training and Testing Score of SVM
# print("Training Score:", sv.score(x_train, y_train))
# print("Testing Score:", sv.score(x_test, y_test))
# Training Score: 0.9741666666666666
# Testing Score: 0.935

#As accuracy of Logistic Regression is higher than SVM , so we will use Logistic Regression

#Training through other ways
#Training Data Accuracy
x_train_prediction = lg.predict(x_train)
acc = accuracy_score(y_train,x_train_prediction)

print (acc)

acc = round(acc*100,2)

print(acc ,'%')

#Testing Data Accuracy

x_test_prediction = lg.predict(x_test)
acctest = accuracy_score(y_test,x_test_prediction)

print (acctest)

acctest = round(acctest*100,2)

print(acctest ,'%')

#Testing Model on an image

test_img = cv2.imread('/Users/dushyantpant/Desktop/Personal Project/Brain_Tumor Detection/backend/pred/pred5.jpg')

img = Image.fromarray(test_img)

img = img.resize((60,60))

img = np.array(img)

img = img.reshape(1,-1)/255

result = lg.predict(img)

result = (dec[result[0]])

print(result)


#Model Building using Pickle

pickle.dump(lg, open('model.pkl', 'wb'))
