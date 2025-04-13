from pymongo import MongoClient
from urllib.parse import quote_plus
import os

def init_mongodb():
    """ 운영 환경에서 직접 MongoDB에 접속 """
    DB_HOST = os.getenv("MONGODB_HOST")
    DB_PORT = os.getenv("MONGODB_PORT")
    DB_USERNAME = quote_plus(os.getenv("MONGODB_USER"))
    DB_PASSWORD = quote_plus(os.getenv("MONGODB_PASSWORD"))
    DB_AUTH_SOURCE = os.getenv("MONGODB_AUTH_SOURCE")

    client = MongoClient(f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/?authSource={DB_AUTH_SOURCE}")

    return client