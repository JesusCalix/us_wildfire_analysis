"""Flask configuration."""
from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

"""Set Flask config variables."""

TESTING = True
DEBUG = True
# FLASK_ENV = environ.get('FLASK_ENV')
# SECRET_KEY = environ.get('SECRET_KEY')
# STATIC_FOLDER = 'static'
# TEMPLATES_FOLDER = 'templates'

# Database
# SQLALCHEMY_DATABASE_URI = environ.get('SQLALCHEMY_DATABASE_URI')
# SQLALCHEMY_TRACK_MODIFICATIONS = False

# AWS Secrets
# AWS_SECRET_KEY = environ.get('AWS_SECRET_KEY')
# AWS_KEY_ID = environ.get('AWS_KEY_ID')