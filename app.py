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

@app.route('/details.html')
def details():
    return render_template('details.html')

@app.route('/previous.html')
def details():
    return render_template('previous.html')

@app.route('/constituencies.html')
def details():
    return render_template('constituencies.html')

@app.route('/compare.html')
def details():
    return render_template('compare.html')
