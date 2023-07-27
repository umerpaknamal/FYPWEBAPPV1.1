import numpy as np
from flask import Flask, request, jsonify, render_template,send_file
import torch
from torchvision import transforms
from PIL import Image
from train import PreTrainedResNet
import urllib
import uuid
import os
app = Flask(__name__)
model = torch.load(open('C:/Users/umar  masood/Favorites/Downloads/Deployment-flask-master/Deployment-flask-master/flask/model.pkl','rb'),map_location=torch.device('cpu'))
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# model = pickle.load(open('C:/Users/umar  masood/Favorites/Downloads/Deployment-flask-master/Deployment-flask-master/model.pkl','rb'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                        "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result = predict(img_path , model)

                predictions = {
                      "class1":class_result
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')



def predict(image_file , model):  

#    model = torch.load(open('C:/Users/umar  masood/Favorites/Downloads/Deployment-flask-master/Deployment-flask-master/flask/model.pkl','rb'),map_location=torch.device('cpu'))
#    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
   # if request.method =='POST':
    #   image_file =  request.files['image']
    #    print(image_file)
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load and preprocess the image
    image = Image.open(image_file)
    #print(image)
    image = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        result = model(image)
    _, predicted = torch.max(result, 1)
    class_index = predicted.item()
    class_label = classes[class_index]
    return class_label




   
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('new.html')


if __name__ == "__main__":

    app.run(debug=True)

