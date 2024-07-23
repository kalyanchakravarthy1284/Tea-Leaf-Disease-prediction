
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask , request, render_template
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model(r"D:\Web Interface\Tea-LeavesDisease-Detection-Model.h5",compile=False)
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img=image.load_img(filepath,target_size=(224,224))
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)
        y=model.predict(x)
        preds=np.argmax(y, axis=1)
        #preds = model.predict_classes(x)
        print("prediction",preds)
        Index = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light',
        'healthy', 'red leaf spot', 'white spot']
        text = "The Detected Disease is : " + str(Index[preds[0]])
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
