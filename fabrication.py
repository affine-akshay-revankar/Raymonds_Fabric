import flask
from flask import Flask
from flask import jsonify
from flask import request
from PIL import Image
import io
import os
from io import BytesIO
import flask
import json
from flask_cors import CORS
import requests
import torch
from fastai import *
from fastai.vision import load_learner
from fastai.vision import open_image


app = Flask(__name__)
CORS(app)
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))


learn = load_learner("",'5D_SIN_Resnet50_SGD.pkl')



@app.route("/predictnewimage", methods=["POST"])
def predictnewimage():
    print('first')
    
    
    resp = {"success": False}
   
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print('second')
        if flask.request.files.get("pic"):
            print("has pic")
            # read the image in PIL format
            img = flask.request.files["pic"].read()
            #print("img>>>>",ig)
            img_open = Image.open(io.BytesIO(img))
            img_open.save("./raw_image.jpg")
            
            
            np_image = open_image("./raw_image.jpg") 


            #resp["predictions"] = []
            resp =[]
            
            resp = learn.predict(np_image)
            print(resp[2])
        
            probability = resp[2].numpy()
            defect_name = ['Broken End','Broken Pick','Missing Pick','No Defect','Rub Mark','Starting Mark'] 

            resp = dict(zip(defect_name, probability))
            print(resp)
            for key in resp:
                resp[key] = round(resp[key]*100 ,2)


            return json.dumps(str(resp))
    else:
        return "Where is the image?"




@app.route("/predict", methods=["POST"])
def predict():
    
    print('first')
    
    # initialize the data dictionary that will be returned from the
    # view
    resp = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print('second')
       
        print("has pic")
        url= request.data
        print(url)
        response = requests.get(url)
        print(response)
       
        img_open = Image.open(BytesIO(response.content))
        img_open.save("./raw_image.jpg")
        
        np_image = open_image("./raw_image.jpg") 
        

        #resp["predictions"] = []
        
        resp =[]

        resp = learn.predict(np_image)
        print(resp[2])

        probability = resp[2].numpy()
        defect_name = ['Broken End','Broken Pick','Missing Pick','No Defect','Rub Mark','Starting Mark'] 

        resp = dict(zip(defect_name, probability))
        print('before multiflication: ',resp)
        for key in resp:
            resp[key] = round(resp[key]*100 ,2)
            
        print('After multiflication :',resp)
            
            
        return json.dumps(str(resp))
    else:
        return "Where is the image?"



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5010, debug=False)
    
