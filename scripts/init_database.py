"""
数据库初始化脚本
创建必要的数据库表和示例数据

使用方法:
    python scripts/init_database.py
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.database import Database
from src.config import DB_PATH


def init_database():
    """初始化数据库"""
    print("=" * 60)
    print("数据库初始化")
    print("=" * 60)
    
    db = Database(DB_PATH)
    
    print(f"数据库路径: {DB_PATH}")
    print("✓ 数据库表已创建/更新")
    
    return db


def add_sample_students(db):
    """添加示例学生数据"""
    print("\n添加示例学生...")
    
    sample_students = [
        ('2024001', '张三'),
        ('2024002', '李四'),
        ('2024003', '王五'),
        ('2024004', '赵六'),
        ('2024005', '孙七'),
    ]
    
    for student_id, name in sample_students:
        db.add_student(student_id, name)
        print(f"  ✓ 添加学生: {name} ({student_id})")
    
    print(f"\n共添加 {len(sample_students)} 名示例学生")


def show_database_info(db):
    """显示数据库信息"""
    print("\n" + "=" * 60)
    print("数据库信息")
    print("=" * 60)
    
    students = db.get_all_students()
    print(f"学生总数: {len(students)}")
    
    if len(students) > 0:
        print("\n学生列表:")
        for s in students:
            has_face = "✓" if s['face_encoding'] is not None else "✗"
            print(f"  [{has_face}] {s['id']} - {s['name']}")
    
    # 考勤记录统计
    records = db.get_attendance_records()
    print(f"\n今日考勤记录: {len(records)}")
    sessions = db.list_attendance_sessions(limit=5)
    print(f"最近场次条数（最多列 5 条）: {len(sessions)}")


def reset_database(db):
    """重置数据库（清空所有数据）"""
    import sqlite3
    
    print("\n警告: 即将清空所有数据!")
    confirm = input("输入 'yes' 确认重置: ")
    
    if confirm.lower() == 'yes':
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM attendance")
        cursor.execute("DELETE FROM attendance_sessions")
        cursor.execute("DELETE FROM students")
        
        conn.commit()
        conn.close()
        
        print("✓ 数据库已重置")
    else:
        print("取消重置")


def main():
    parser = argparse.ArgumentParser(description='初始化数据库')
    parser.add_argument('--sample', action='store_true',
                        help='添加示例学生数据')
    parser.add_argument('--reset', action='store_true',
                        help='重置数据库（删除所有数据）')
    parser.add_argument('--info', action='store_true',
                        help='显示数据库信息')
    
    args = parser.parse_args()
    
    # 初始化数据库
    db = init_database()
    
    # 处理参数
    if args.reset:
        reset_database(db)
        return
    
    if args.sample:
        add_sample_students(db)
    
    if args.info or not args.sample:
        show_database_info(db)
    
    print("\n" + "=" * 60)
    print("操作完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
