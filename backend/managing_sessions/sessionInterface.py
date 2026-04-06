from connector import Connector
from uuid import uuid4
import json
class SessionInterface:
    def __init__(self):
        self.connector = Connector()
    def check_session_exists(self, session_id):
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            cursor.execute(f"SELECT id FROM sessions WHERE id = {session_id}")
            result = cursor.fetchone()
            if result:
                print("Session exists.")
                return json.dumps({"status": "200", "message": "Session exists."}), result[0]
            else:
                print("Session does not exist.")
                return json.dumps({"status": "404", "message": "Session does not exist."})
        except Exception as e:
            print(f"Error checking session existence: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while checking session existence."})
    def create_session(self, user_id,date_time,title):
        try:
            connection = self.connector.connect()
            cursor = connection.cursor()
            session, session_id = self.check_session_exists(user_id)
            if session and json.loads(session)["status"] == "200":
                print("Session already exists.")
                return json.dumps({"status": "400", "message": "Session already exists."})
            session_id = str(uuid4())
            cursor.execute("INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)", (session_id, user_id, date_time, title))
            connection.commit()
            cursor.close()
            connection.close()
            print("Session created successfully!")
            return json.dumps({"status": "202", "message": "Session created successfully!", "session_id": session_id})
        except Exception as e:
            print(f"Error creating session: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while creating the session"})
    def update_session(self, session_id):
        try:
            connection  = self.connector.connect()
            cursor = connection.cursor()

            cursor.execute("UPDATE sessions SET updated_at = NOW() WHERE id = %s", (session_id,))
            connection.commit()
            cursor.close()
            connection.close()
            print("Session updated successfully!")
            return json.dumps({"status": "200", "message": "Session updated successfully!"})
        except Exception as e:
            print(f"Error updating session: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while updating the session"})
    def get_messages(self,session_id):
        try:
            connection = self.connector.connect()
            cursor = connection.cursor()
            query = f"""
                    SELECT content, role, created_at
                    FROM messages m
                    join sessions s on m.session_id = s.id
                    WHERE s.id = {session_id}
                    ORDER BY m.created_at ASC
                    """
            cursor.execute(query)
            messages = cursor.fetchall()
            cursor.close()
            connection.close()
            messages_list = [{"content": msg[0], "role": msg[1], "created_at": msg[2].isoformat()} for msg in messages]
            return json.dumps({"status": "200", "messages": messages_list})
        except Exception as e:
            print(f"Error fetching messages: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while fetching messages"})