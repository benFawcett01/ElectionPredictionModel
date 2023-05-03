from flask import Flask, render_template, send_file
from model import Model
import time
import pandas as pd

app = Flask(__name__)

ran = False
winner = ''
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/index.html")
def back():
    return render_template('index.html')

@app.route('/results.html')
def results():
    global ran
    global winner
    if ran == False:
        start = time.time()
        m = Model()
        m.run_pop_forecast()
        winner = m.forecast_seats()
        end = time.time()
        ran = True

    return render_template('results.html', w = winner)

    print('Model Runtime: '+str(end-start), "s")


@app.route('/details.html')
def details():
    return render_template('details.html')

@app.route('/previous.html')
def previous():
    return render_template('previous.html')

@app.route('/download')
def download():
    path = 'datasets/past_model_results.csv'
    return send_file(path, as_attachment=True)


@app.route('/constituencies.html')
def constituencies():
    return render_template('constituencies.html')

@app.route('/compare.html')
def compare():
    m = Model()
    m.compare()

    data = pd.read_csv('datasets/comparisons-2019.csv', index_col='model')
    return render_template('compare.html', tables=[data.to_html()], titles=[''])

@app.route('/individual_seat.html')
def individual_seat():
    return render_template('individual_seat.html')


if __name__ == '__main__':
    app.run(debug=True)