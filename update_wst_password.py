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

# 查询wst用户
cursor.execute("SELECT id, username, password FROM user WHERE username = 'wst'")
user = cursor.fetchone()

if user:
    user_id, username, password = user
    print(f"找到用户: {username} (ID: {user_id})")
    
    # 设置新密码
    new_password = 'quickform'
    hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
    
    # 更新密码
    cursor.execute("UPDATE user SET password = ? WHERE id = ?", (hashed_password, user_id))
    conn.commit()
    
    print(f"用户 {username} 的密码已成功更新为 'quickform'")
    print(f"新的密码哈希: {hashed_password}")
else:
    print("未找到用户 'wst'")

# 关闭连接
conn.close()

print("\n密码更新操作完成！")
