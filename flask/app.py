#!/usr/bin/env python
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import datadive as dd

app = Flask(__name__)
CORS(app, origins="*")  # '*' for development stage

if not os.path.isdir("uploads"):
    os.mkdir("uploads")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return "Working correctly!"


@app.route('/upload', methods=['PUT', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        dt = dd.read_csv(filename)

        return jsonify({'success': 'File uploaded successfully', 'file_path': filename, 'data': dt.to_html()})


@app.route('/columns', methods=['POST'])
def get_columns():
    dt = dd.read_html(str(request.data))
    columns = dt.get_columns().tolist()
    print(columns)
    return jsonify(data=columns)


@app.route('/select_columns', methods=['POST'])
def select_columns_handler():
    cols = request.get_json().get("cols")
    html = request.get_json().get("html")[0]

    dt = dd.read_html(html)
    dt = dt.select_columns(cols)

    return jsonify(data=dt.to_html())


@app.route('/get_operators', methods=['POST'])
def get_operators():
    col = request.get_json().get("col")
    html = request.get_json().get("html")[0]

    dt = dd.read_html(html)
    col_type = dd.get_column_type(dt, col)
    operators = dd.get_related_condition_types(col_type)

    return jsonify({
        "col": col,
        "data": operators
    })


@app.route('/select_rows', methods=['POST'])
def select_rows_handler():
    conditions = request.get_json().get("conditions")
    html = request.get_json().get("html")[0]
    dt = dd.read_html(html)
    dt = dd.select_rows(dt, conditions)
    return jsonify(data=dt.to_html())


if __name__ == "__main__":
    app.run("localhost", 5000, debug=True)
