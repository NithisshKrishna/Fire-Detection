import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import cv2
import os
from flask import Flask, render_template, request, redirect, url_for


from werkzeug.utils import secure_filename
from plotly.subplots import make_subplots
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import load_model

from keras.models import load_model
model12 = load_model("network.h5")

def ValuePredict(data):
 
    n="F://Fire_detection//static//uploads" + data
    cam = cv2.VideoCapture(n)
  
    try:
      
    # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
  
# if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
  
# frame
    print("HI")
    currentframe = 0
    temp=0
    cf=0
    while(temp==0 and cf==0):
      
    # reading from frame
        ret,frame = cam.read()
  
        if ret:
        # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
  
        # writing the extracted images
            cv2.imwrite(name, frame)
        
        
        # increasing counter so that it will
        # show how many frames are created
            currentframe += 1
            if(currentframe>200):
                cf=1
            img = image.load_img(name)
            img = image.img_to_array(img)/255
            img = tf.image.resize(img,(256,256))
            img = tf.expand_dims(img,axis=0)
            prediction = int(tf.round(model12.predict(x=img)).numpy()[0][0])
        #print("The predicted value is: ",prediction,"and the predicted label is:",class_indices[prediction])
        #print("Image Shape",img.shape)
        #prediction=0
            print(prediction)
            if(prediction==0):
                temp=1
                print(name)
                print("Fire!!")
                break
        else:
            break
    if(temp==0):
        print("Non-Fire!")        
    #print(temp)
    return(temp)

UPLOAD_FOLDER = 'static/uploads/'
app = Flask( __name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('untitled.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

# @app.route('/result', methods = ['POST'])
# def upload_video():
#     if request.method == 'POST':
#         file = request.files['filename']
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         #print(data)
#         #to_predict_list = request.form.to_dict()
#         #to_predict_list = list(to_predict_list.values())
#         #to_predict_list = list(map(int, to_predict_list))
#         result = ValuePredict(filename)   
#         print("PREDICTED")
#         print('upload_video filename: ' + filename)
#         print(result)
#         a=""
#         if int(result)== 1:
#              a ='FIRE'
#         else:
#             a="NON - FIRE "           
#         return render_template("result.html", prediction = a, filena=filename)

@app.route('/result', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['filename']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = ValuePredict(filename)
            print("PREDICTED")
            print('upload_image filename: ' + filename)
            print(result)
            a = ""
            if int(result) == 1:
                a = 'FIRE'
            else:
                a = "NON-FIRE"
            return render_template("result.html", prediction=a, filena=filename)


@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    
    if prediction[0]==1:
        return render_template('index.html',
                               prediction_text='Extinguished'.format(prediction),
                               )
    elif prediction[0]==0:
        return render_template('index.html',
                               prediction_text='Not extinguished'.format(prediction),
                              )
    
app.run(debug=True, use_reloader=False)