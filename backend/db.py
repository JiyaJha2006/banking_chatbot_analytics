import mysql.connector


def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="banking_chatbot_db"
    )