from connector import Connector
from uuid import uuid4
from bcrypt import hashpw, gensalt
import json
SALT = gensalt()
class UserInterface:
    def __init__(self):
        self.connector = Connector()
    def verify_user(self,email,password):
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            cursor.execute(f"SELECT password FROM users WHERE email = {email}")
            result = cursor.fetchone()
            if result:
                stored_password = result[0]
                if hashpw(password.encode('utf-8'), stored_password.encode('utf-8')) == stored_password.encode('utf-8'):
                    print("User verified successfully!")
                    return json.dumps({"status": "200", "message": "User verified successfully!"})
                else:
                    print("Incorrect password.")
                    return json.dumps({"status": "400", "message": "Incorrect password."})
            else:
                print("User not found.")
                return json.dumps({"status": "404", "message": "User not found."})
        except Exception as e:
            print(f"Error verifying user: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while verifying the user."})
    def check_user_exists(self,email):
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            cursor.execute(f"SELECT id FROM users WHERE email = {email}")
            result = cursor.fetchone()
            if result:
                print("User exists.")
                return json.dumps({"status": "200", "message": "User exists."})
            else:
                print("User does not exist.")
                return json.dumps({"status": "404", "message": "User does not exist."})
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while checking user existence."})
    def add_user(self,username,email,password):
        try:
            isuser = self.check_user_exists(email)
            if isuser and json.loads(isuser)["status"] == "200":
                print("User already exists.")
                return json.dumps({"status": "400", "message": "User already exists."})
            elif isuser and json.loads(isuser)["status"] == "500":
                print("Error checking user existence.")
                return json.dumps({"status": "500", "message": "An error occurred while checking user existence."})
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            user_id = str(uuid4())
            hashed_password = hashpw(password.encode('utf-8'), SALT)
            cursor.execute("INSERT INTO users (id, username, email, password) VALUES (%s, %s, %s, %s)", (user_id, username, email, hashed_password))
            connection.commit()
            print(f"User {username} added successfully with ID: {user_id}")
            return json.dumps({"status": "202", "message": f"User {username} added successfully with ID: {user_id}"})
        except Exception as e:
            print(f"Error adding user: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while adding the user."})
    
    def get_user_id(self,email):
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            cursor.execute(f"SELECT id FROM users WHERE email = {email}")
            result = cursor.fetchone()
            if result:
                user_id = result[0]
                print(f"User ID for email {email} is {user_id}.")
                return json.dumps({"status": "200", "message": f"User ID for email {email} is {user_id}.", "user_id": user_id})
            else:
                print("User not found.")
                return json.dumps({"status": "404", "message": "User not found."})
        except Exception as e:
            print(f"Error fetching user ID: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while fetching the user ID."})

