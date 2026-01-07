import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime

# 连接到SQLite数据库
conn = sqlite3.connect('quickform.db')
cursor = conn.cursor()

print("正在清理数据库中的用户和API密钥信息...")

# 删除现有的AI配置信息
try:
    cursor.execute("DELETE FROM ai_config")
    print("已删除所有AI配置信息")
except sqlite3.Error as e:
    print(f"删除AI配置信息时出错: {e}")

# 删除现有的用户信息
try:
    cursor.execute("DELETE FROM user")
    print("已删除所有用户信息")
except sqlite3.Error as e:
    print(f"删除用户信息时出错: {e}")

# 添加管理员用户
print("\n正在添加管理员用户...")
try:
    # 生成密码哈希
    hashed_password = generate_password_hash('quickform')
    
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 插入管理员用户
    cursor.execute(
        "INSERT INTO user (username, email, password, created_at) VALUES (?, ?, ?, ?)",
        ('wst', 'admin@quickform.com', hashed_password, current_time)
    )
    print("管理员用户添加成功！")
    print("用户名: wst")
    print("密码: quickform")
except sqlite3.Error as e:
    print(f"添加管理员用户时出错: {e}")

# 提交更改
conn.commit()

# 关闭连接
conn.close()

print("\n数据库重置完成！")
