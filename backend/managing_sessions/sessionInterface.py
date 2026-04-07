from connector import Connector
from uuid import uuid4
import json
class SessionInterface:
    def __init__(self):
        self.connector = Connector()
    def set_title(self, session_id, title):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            cursor = connection.cursor()
            query = f"UPDATE session SET title = {connection.escape(title)}, updated_at = NOW() WHERE id = {connection.escape(session_id)}"
            cursor.execute(query)
            connection.commit()
            print("Session title updated successfully!")
            return json.dumps({"status": "200", "message": "Session title updated successfully!"})
        except Exception as e:
            print(f"Error updating session title: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while updating the session title"})
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def get_session(self, user_id):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            query = (
                f"SELECT id, title, updated_at "
                f"FROM session "
                f"WHERE user_id = {connection.escape(user_id)} "
                f"ORDER BY updated_at DESC"
            )
            cursor.execute(query)
            results = cursor.fetchall()
            if results:
                sessions = []
                for row in results:
                    updated_at = row[2].isoformat() if row[2] else None
                    sessions.append(
                        {
                            "session_id": row[0],
                            "title": row[1] or "Chat session",
                            "updated_at": updated_at,
                        }
                    )

                print("Sessions fetched.")
                return (
                    json.dumps(
                        {
                            "status": "200",
                            "message": "Sessions fetched.",
                            "sessions": sessions,
                        }
                    ),
                    sessions,
                )
            else:
                print("Session does not exist.")
                return json.dumps({"status": "404", "message": "Session does not exist."}), None
        except Exception as e:
            print(f"Error checking session existence: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while checking session existence."}), None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def check_session_exists(self, session_id):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            print("Connection successful!")
            cursor = connection.cursor()
            query = f"SELECT id FROM session WHERE id = {connection.escape(session_id)}"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                print("Session exists.")
                return json.dumps({"status": "200", "message": "Session exists."}), result[0]
            else:
                print("Session does not exist.")
                return json.dumps({"status": "404", "message": "Session does not exist."}), None
        except Exception as e:
            print(f"Error checking session existence: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while checking session existence."}), None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def create_session(self, user_id,date_time,title):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            cursor = connection.cursor()
            session, _existing_session_id = self.check_session_exists(user_id)
            if session and json.loads(session)["status"] == "200":
                print("Session already exists.")
                return json.dumps({"status": "400", "message": "Session already exists."})
            session_id = str(uuid4())
            query = (
                f"INSERT INTO session (id, user_id, title, created_at, updated_at) VALUES ("
                f"{connection.escape(session_id)}, {connection.escape(user_id)}, "
                f"{connection.escape(title)}, {connection.escape(date_time)}, {connection.escape(date_time)})"
            )
            cursor.execute(query)
            connection.commit()
            print("Session created successfully!")
            return json.dumps({"status": "202", "message": "Session created successfully!", "session_id": session_id})
        except Exception as e:
            if connection:
                connection.rollback()
            print(f"Error creating session: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while creating the session"})
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def update_session(self, session_id):
        connection = None
        cursor = None
        try:
            connection  = self.connector.connect()
            cursor = connection.cursor()

            query = f"UPDATE session SET updated_at = NOW() WHERE id = {connection.escape(session_id)}"
            cursor.execute(query)
            connection.commit()
            print("Session updated successfully!")
            return json.dumps({"status": "200", "message": "Session updated successfully!"})
        except Exception as e:
            print(f"Error updating session: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while updating the session"})
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def get_messages(self,session_id):
        connection = None
        cursor = None
        try:
            connection = self.connector.connect()
            cursor = connection.cursor()
            query = f"""
                    SELECT m.content, m.role, m.created_at
                    FROM messages m
                    join session s on m.session_id = s.id
                WHERE s.id = {connection.escape(session_id)}
                    ORDER BY m.created_at ASC
                    """
            cursor.execute(query)
            messages = cursor.fetchall()
            messages_list = [{"content": msg[0], "role": msg[1], "created_at": msg[2].isoformat()} for msg in messages]
            return json.dumps({"status": "200", "messages": messages_list})
        except Exception as e:
            print(f"Error fetching messages: {e}")
            return json.dumps({"status": "500", "message": "An error occurred while fetching messages"})
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()