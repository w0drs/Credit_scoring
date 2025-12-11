from src.backend.database import Database

def test_database():
    db = Database("../../data/database/applications.db")
    assert db
    client = db.get_client_by_id(123)
    assert client