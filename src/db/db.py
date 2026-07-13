from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_DIR = BASE_DIR / "data/database"
DB_PATH = DB_DIR / "database.db"

DB_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

print(f"Подключение к БД: {DATABASE_URL}")

engine = create_async_engine(DATABASE_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_async_session():
    async with async_session_maker() as session:
        yield session
