from app import app
from flask import Flask, render_template, request

import Running

@app.route('/test', methods = ["GET", "POST"])
def test():
    if request.method == "POST":
       path = request.form.get("path")
       return "You Entered: " + path
    return render_template('test.html', title='File Classifier')

@app.route('/', methods = ["GET", "POST"])
def fileClassifier():
    issuccess = True
    if request.method == "POST":
        path = request.form.get("path")
        targetPath = request.form.get("targetPath")
        model = Running.Classifier(path, targetPath)
        issuccess = model.classify()
        
        if issuccess:
            return render_template('success.html', title='File Classifier', path = path, issuccess = issuccess)
        else:
            return render_template('fileClassifier.html', title='File Classifier', issuccess = issuccess)
    return render_template('fileClassifier.html', title='File Classifier', issuccess = issuccess)
