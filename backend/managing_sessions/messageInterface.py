from connector import Connector
from uuid import uuid4
import json

class MessageInterface:
    def __init__(self):
        self.connector = Connector()

    def send_message(self, session_id, content, role):
        connection = None
        cursor = None
        try:
            message_id = str(uuid4())
            connection = self.connector.connect()
            cursor = connection.cursor()
            insert_query = (
                f"INSERT INTO messages (id, session_id, content, role, created_at) VALUES ("
                f"{connection.escape(message_id)}, {connection.escape(session_id)}, "
                f"{connection.escape(content)}, {connection.escape(role)}, NOW())"
            )
            cursor.execute(insert_query)

            update_query = f"UPDATE session SET updated_at = NOW() WHERE id = {connection.escape(session_id)}"
            cursor.execute(update_query)

            connection.commit()
            return json.dumps({"status": "200", "message_id": message_id})
        except Exception as e:
            if connection:
                connection.rollback()
            return json.dumps({"status": "500", "error": str(e)})
        finally:
            if cursor:
                cursor.close()
            if connection:
                try:
                    connection.close()
                except Exception:
                    pass
        
   