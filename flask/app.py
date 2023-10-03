#!/usr/bin/env python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*") # '*' for development stage


@app.route("/")
def home():
    return "Working correctly!"


if __name__ == "__main__":
    app.run("localhost", 5000, debug=True)
