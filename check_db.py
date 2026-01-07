import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String

# 创建数据库连接
DATABASE_URL = 'sqlite:///quickform.db'
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义用户模型（简化版本，只包含必要字段）
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(200), nullable=False)

def check_user_passwords():
    db = SessionLocal()
    try:
        # 查询所有用户
        users = db.query(User).all()
        print(f"Found {len(users)} users in the database")
        
        for user in users:
            print(f"User ID: {user.id}, Username: {user.username}")
            print(f"Password hash length: {len(user.password)}")
            print(f"Password hash: {user.password}")
            print("-" * 50)
            
            # 检查密码哈希格式是否正确（Flask-Bcrypt的哈希通常以$2b$或$2a$开头）
            if not user.password.startswith(('$2b$', '$2a$')):
                print(f"WARNING: Invalid password hash format for user {user.username}")
                print("-" * 50)
                
    finally:
        db.close()

if __name__ == '__main__':
    check_user_passwords()