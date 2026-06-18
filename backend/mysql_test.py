from db import get_mysql_connection

conn = get_mysql_connection()

print("MySQL connected successfully!")

conn.close()