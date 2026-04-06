from connector import Connector
from uuid import uuid4
import json

class MessageInterface:
    def __init__(self):
        self.connector = Connector()
    