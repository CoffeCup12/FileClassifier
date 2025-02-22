from app import app
from flask import Flask, render_template, request, redirect, url_for

@app.route('/test', methods = ["GET", "POST"])
def test():
    if request.method == "POST":
       path = request.form.get("path")
       return "You Entered: " + path
    return render_template('test.html', title='File Classifier')