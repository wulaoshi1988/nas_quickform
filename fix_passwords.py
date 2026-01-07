import os
import sqlite3
from flask_bcrypt import Bcrypt
from flask import Flask

# 创建Flask应用实例
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_secret_key')

# 初始化Bcrypt
bcrypt = Bcrypt(app)

# 连接到数据库
conn = sqlite3.connect('quickform.db')
cursor = conn.cursor()

# 查询所有用户
cursor.execute("SELECT id, username, password FROM user")
users = cursor.fetchall()

print("检查用户密码格式...")
for user in users:
    user_id, username, password = user
    print(f"用户 {username} (ID: {user_id}) 密码长度: {len(password)}, 前10个字符: {password[:10]}...")
    
    # 检查密码是否是有效的bcrypt哈希
    if len(password) < 60 or not password.startswith('$2b$'):
        print(f"用户 {username} 的密码格式不正确，需要重置")
        # 重置密码为默认值 'password123'
        new_password = 'password123'
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        cursor.execute("UPDATE user SET password = ? WHERE id = ?", (hashed_password, user_id))
        print(f"用户 {username} 的密码已重置为 'password123'")
    else:
        print(f"用户 {username} 的密码格式正确")

# 提交更改并关闭连接
conn.commit()
conn.close()

print("\n密码检查和修复完成！")
print("使用默认密码 'password123' 登录已修复的账户")
