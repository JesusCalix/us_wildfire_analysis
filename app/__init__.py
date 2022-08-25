from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)
app.config.from_pyfile('config.py')

from app.views import views