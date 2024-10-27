import time
import queue
import asyncio
import threading
from mongoengine import Document, EmbeddedDocument, fields
from loguru import logger
from pymongo.errors import PyMongoError
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection


class SingletonMeta(type):
    """
    Singeleton class for managing single connection.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Special method to which is acting as the **`Heart`** of singleton implementation.

        Returns:
        Existing instance if it exists, else creates a new instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Vehicle(EmbeddedDocument):
    vehicle_id = fields.IntField()
    vehicle_type = fields.IntField()


class VehiclesDOM(Document):
    timestamp = fields.IntField()
    lane = fields.StringField()
    vehicles = fields.ListField(fields.EmbeddedDocumentField(Vehicle))


class MongoClient(metaclass=SingletonMeta):
    """
    Class for managing MongoDB client.
    """

    def __init__(self):
        """
        Initialize MongoClient class.
        """
        self.mongo_url = "mongodb://localhost:27017"
        self.mongo_db = "vehicles"
        self.mongo_collection = "Mawa"
        self.client = None

    def get_client(self) -> AsyncIOMotorCollection:
        """
        Return MongoDB client.

        Returns:
            AsyncIOMotorCollection: MongoDB client.
        """
        try:
            self.client = AsyncIOMotorClient(self.mongo_url)
            database = self.client.get_database(self.mongo_db)
            collection = database.get_collection(self.mongo_collection)

            return collection
        except PyMongoError as e:
            logger.error(f"MongoDB client error: {repr(e)}")
            time.sleep(1)
            return self.get_client()

    def disconnect(self) -> None:
        """
        Disconnect MongoDB client.
        """
        if self.client is not None:
            self.client.close()

    def retry(self) -> AsyncIOMotorCollection:
        """
        Retries MongoDB client collection by disconnecting first.

        Returns:
            AsyncIOMotorCollection: MongoDB client collection.
        """
        self.disconnect()
        self.__init__()
        return self.get_client()


class MongoPublisher(threading.Thread):
    """
    Publishes results to MongoDB collection.
    """

    def __init__(self):
        """
        Initialise the MongoPublisher with MongoClient instance.
        """
        super().__init__()
        self.mongo_client = MongoClient().get_client()
        self.send_queue = queue.Queue()
        self.retries: int = 10
        self.retry_delay: int = 1
        self.loop = None
        self.daemon = True
        self.previous_data = []
        self.start()

    def __process_and_publish(self, _time, _lane, _vehicles: list, coords: list) -> None:
        """
        Processes the received data and publishes it to MongoDB collection.

        Args:
            _time (int): Time in milliseconds.
            _lane (int): Lane number.
            _vehicles (list): List of vehicles.
        """
        x_min, y_min, x_max, y_max = coords
        for attempt in range(self.retries):
            try:
                raw_vehicles = [item for item in _vehicles if len(item) >= 7]
                filtered_vehicles = [
                    [id_, class_] for x1, y1, x2, y2, id_, conf_, class_ in raw_vehicles
                    if (y_min <= (y1 + y2) / 2)
                ]

                if len(filtered_vehicles) >= 1 and filtered_vehicles != self.previous_data:
                    self.previous_data = filtered_vehicles
                    _vehicles = [Vehicle(vehicle_id=data[0], vehicle_type=data[1]) for data in filtered_vehicles]

                    self.loop.run_until_complete(
                        self.mongo_client.insert_one(
                            VehiclesDOM(
                                timestamp=_time,
                                lane=_lane,
                                vehicles=_vehicles
                            ).to_mongo()
                        )
                    )
                    break
            except PyMongoError as e:
                logger.exception(f"Error while inserting data to MongoDB: {repr(e)}")
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.exception(f"Could not insert data to MongoDB: {repr(e)}")

    def publish(self, **kwargs) -> None:
        """
        Puts data on the Queue to be processed and published to MongoDB collection.
        """
        self.send_queue.put(lambda: self.__process_and_publish(**kwargs))

    def run(self):
        """
        Runs the operations on a separate thread.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            while True:
                task = self.send_queue.get()
                task()
        finally:
            self.loop.close()
