from flask import Flask, render_template


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/index.html")
def back():
    return render_template('index.html')

@app.route("/results.html")
def results():
    return render_template('results.html')
