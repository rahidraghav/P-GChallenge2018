from flask import Flask,flash,abort,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
from pymongo import MongoClient

import os
import logging


myclient = MongoClient("mongodb://localhost:27017/")
mydb = myclient["TESTDB1"]
mycol = mydb["lightandstep"]


app = Flask(__name__)
logger=logging.getLogger(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
UPLOAD_FOLDER= './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return redirect(url_for('hello'))

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name = None):
    return render_template('hello.html',name=name)


@app.route('/lightandstep',methods=['POST'])
def lightandstep():
    req_data = request.get_json()
    _lightData = req_data['light']
    _stepData = req_data['step']
    
    post_data = {
        'light': _lightData,
        'step': _stepData
    }
    
    result = mycol.insert_one(post_data)
    
    return '''
              The result is : {}
           '''.format(result)



@app.route('/upload/<filename>',methods = ['GET'])
@app.route('/upload/',methods = ['GET','POST'])
def upload_file(f=None):
      if request.method=='GET':
         return render_template('ImageUpload.html')
      elif  request.method =='POST':
         file = request.files['file[]']
         if file and allowed_file(file.filename):
            f = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],f))
            ambientLight=request.form['inputambientlight']
            inputSteps=request.form['inputsteps']
            print(ambientLight,inputSteps)
            return render_template('ImageUpload.html',filename=f)
        

if __name__ == '__main__':
    app.run(debug = True)