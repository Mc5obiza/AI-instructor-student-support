from connector import Connector
from uuid import uuid4
from bcrypt import hashpw, gensalt
import json
SALT = gensalt()
class UserInterface:
    def __init__(self):
        self.connector = Connector()

    def verify_user(self,email,password):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            query = f"SELECT password FROM users WHERE email = {connection.escape(email)}"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                stored_password = result[0]
                if isinstance(stored_password, bytes):
                    stored_hash = stored_password
                else:
                    stored_hash = str(stored_password).encode('utf-8')

                if hashpw(password.encode('utf-8'), stored_hash) == stored_hash:
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
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def check_user_exists(self,email):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            query = f"SELECT id FROM users WHERE email = {connection.escape(email)}"
            cursor.execute(query)
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
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def add_user(self,username,email,password):
        connection = None
        cursor = None
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
            query = (
                f"INSERT INTO users (id, name, email, password) VALUES ("
                f"{connection.escape(user_id)}, {connection.escape(username)}, "
                f"{connection.escape(email)}, {connection.escape(hashed_password)})"
            )
            cursor.execute(query)
            connection.commit()
            print(f"User {username} added successfully with ID: {user_id}")
            return json.dumps({"status": "202", "message": f"User {username} added successfully with ID: {user_id}"})
        except Exception as e:
            if connection:
                connection.rollback()
            print(f"Error adding user: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while adding the user."})
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def get_user_id(self,email):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            query = f"SELECT id FROM users WHERE email = {connection.escape(email)}"
            cursor.execute(query)
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
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

