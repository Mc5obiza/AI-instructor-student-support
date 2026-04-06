from __future__ import annotations
from numpy import rint
import pymysql

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "config" / "DB_CONFIG.py").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.DB_CONFIG import DB_CONFIG

class Connector:
    def __init__(self):
        self.__host = DB_CONFIG["HOST"]
        self.__user = DB_CONFIG["USER"]
        self.__password = DB_CONFIG["PASSWORD"]
        self.__database = DB_CONFIG["DATABASE"]
        self.__port = DB_CONFIG["PORT"]
    def display_tables(self):
        try:
            connection = self.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            cursor.execute("Show tables")
            tables = cursor.fetchall()
            print(f"Tables in the database: {tables}")
        except Exception as e:
            print(f"Error fetching tables: {e}")

    def connect(self):
        # Implement the logic to connect to the database using the above credentials
        
            return pymysql.connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            )
    def fetch_tables(self,connection):
        # Implement the logic to fetch tables from the database
        if connection:
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            return tables
        else:
            print("No active database connection.")
            return None

        
    def disconnect(self, connection):
        # Implement the logic to disconnect from the database
        if connection:
            connection.close()
            print("Database connection closed.")
    
if __name__ == "__main__":
    connector = Connector()
    try:
        connection = connector.connect()
        print("Connection successful!")
        cursor = connection.cursor()
        cursor.execute("Show tables")
        tables = cursor.fetchall()
        print(f"Tables in the database: {tables}")
    except Exception as e:
        print(f"Connection failed: {e}")
  
    
    