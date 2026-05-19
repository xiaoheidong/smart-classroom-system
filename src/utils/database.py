"""
数据库模块
管理学生信息、考勤记录
"""
import sqlite3
import pickle
import numpy as np
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict
from pathlib import Path


class Database:
    """SQLite数据库管理类"""
    
    def __init__(self, db_path: str):
        """
        初始化数据库
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 学生信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_encoding BLOB,
                register_time TIMESTAMP
            )
        ''')
        
        # 考勤场次：每次「开始点名」为一堂课的签到窗口，「停止」时结束
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP,
                ended_at TIMESTAMP
            )
        ''')
        
        # 考勤记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                check_in_time TIMESTAMP,
                confidence REAL,
                session_id INTEGER,
                FOREIGN KEY (student_id) REFERENCES students (id),
                FOREIGN KEY (session_id) REFERENCES attendance_sessions (id)
            )
        ''')
        
        # 作弊检测页：异常行为抓拍存证（图片路径 + 元数据）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                behavior_type TEXT NOT NULL,
                track_id INTEGER,
                image_path TEXT NOT NULL,
                confidence REAL,
                trigger_reason TEXT,
                detail_json TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        self._migrate_attendance_session_column()
    
    def _migrate_attendance_session_column(self):
        """旧库无 session_id 时补列并确保场次表存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(attendance)")
        cols = [row[1] for row in cursor.fetchall()]
        if "session_id" not in cols:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP
                )
            ''')
            cursor.execute(
                "ALTER TABLE attendance ADD COLUMN session_id INTEGER "
                "REFERENCES attendance_sessions (id)"
            )
        conn.commit()
        conn.close()
    
    def add_student(self, 
                    student_id: str, 
                    name: str, 
                    face_encoding: Optional[np.ndarray] = None) -> bool:
        """
        添加学生
        Args:
            student_id: 学号
            name: 姓名
            face_encoding: 人脸特征向量（128维）
        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            encoding_blob = None
            if face_encoding is not None:
                encoding_blob = pickle.dumps(face_encoding)
            
            # 使用本地时间
            local_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT OR REPLACE INTO students (id, name, face_encoding, register_time)
                VALUES (?, ?, ?, ?)
            ''', (student_id, name, encoding_blob, local_now))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"添加学生失败: {e}")
            return False
    
    def get_student(self, student_id: str) -> Optional[Dict]:
        """
        获取学生信息
        Args:
            student_id: 学号
        Returns:
            学生信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, face_encoding, register_time
            FROM students WHERE id = ?
        ''', (student_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'name': result[1],
                'face_encoding': pickle.loads(result[2]) if result[2] else None,
                'register_time': result[3]
            }
        return None
    
    def get_all_students(self) -> List[Dict]:
        """
        获取所有学生
        Returns:
            学生信息列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, face_encoding, register_time FROM students
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        students = []
        for result in results:
            students.append({
                'id': result[0],
                'name': result[1],
                'face_encoding': pickle.loads(result[2]) if result[2] else None,
                'register_time': result[3]
            })
        return students
    
    def delete_student(self, student_id: str) -> bool:
        """
        删除学生
        Args:
            student_id: 学号
        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM students WHERE id = ?', (student_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"删除学生失败: {e}")
            return False
    
    def start_attendance_session(self) -> int:
        """开始一堂课的签到，返回 session_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 使用本地时间而非 UTC 时间
        local_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            "INSERT INTO attendance_sessions (started_at, ended_at) VALUES (?, NULL)",
            (local_now,)
        )
        sid = cursor.lastrowid
        conn.commit()
        conn.close()
        return sid
    
    def end_attendance_session(self, session_id: int) -> bool:
        """结束本场签到（签到时间截止）"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # 使用本地时间而非 UTC 时间
            local_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                '''
                UPDATE attendance_sessions
                SET ended_at = ?
                WHERE id = ? AND ended_at IS NULL
                ''',
                (local_now, session_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"结束场次失败: {e}")
            return False
    
    def record_attendance(
        self,
        student_id: str,
        confidence: float,
        session_id: Optional[int] = None,
    ) -> bool:
        """
        记录考勤（绑定到场次；同一场内同一学生只记一次）
        Args:
            student_id: 学号
            confidence: 识别置信度
            session_id: 本场签到 id；为 None 时兼容旧逻辑（按自然日去重，无 session）
        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 使用本地时间
            local_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if session_id is not None:
                cursor.execute(
                    '''
                    SELECT COUNT(*) FROM attendance
                    WHERE student_id = ? AND session_id = ?
                    ''',
                    (student_id, session_id),
                )
                if cursor.fetchone()[0] > 0:
                    conn.close()
                    return True
                cursor.execute(
                    '''
                    INSERT INTO attendance (student_id, confidence, session_id, check_in_time)
                    VALUES (?, ?, ?, ?)
                    ''',
                    (student_id, confidence, session_id, local_now),
                )
            else:
                today = datetime.now().strftime("%Y-%m-%d")
                cursor.execute(
                    '''
                    SELECT COUNT(*) FROM attendance
                    WHERE student_id = ? AND DATE(check_in_time) = ?
                      AND session_id IS NULL
                    ''',
                    (student_id, today),
                )
                if cursor.fetchone()[0] > 0:
                    conn.close()
                    return True
                cursor.execute(
                    '''
                    INSERT INTO attendance (student_id, confidence, session_id, check_in_time)
                    VALUES (?, ?, NULL, ?)
                    ''',
                    (student_id, confidence, local_now),
                )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"记录考勤失败: {e}")
            return False
    
    def get_attendance_records(self, 
                               date: Optional[str] = None) -> List[Dict]:
        """
        获取考勤记录
        Args:
            date: 日期字符串（YYYY-MM-DD），默认今天
        Returns:
            考勤记录列表
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.id, a.student_id, s.name, a.check_in_time, a.confidence
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE DATE(a.check_in_time) = ?
            ORDER BY a.check_in_time DESC
        ''', (date,))
        
        results = cursor.fetchall()
        conn.close()
        
        records = []
        for result in results:
            records.append({
                'id': result[0],
                'student_id': result[1],
                'name': result[2],
                'check_in_time': result[3],
                'confidence': result[4]
            })
        return records
    
    def get_today_attendance_summary(self) -> Tuple[List[str], List[str]]:
        """
        获取今日考勤汇总
        Returns:
            (已签到学生列表, 未到学生列表)
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取已签到学生
        cursor.execute('''
            SELECT DISTINCT s.name
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE DATE(a.check_in_time) = ?
        ''', (today,))
        
        present = [row[0] for row in cursor.fetchall()]
        
        # 获取未到学生
        cursor.execute('''
            SELECT name FROM students
            WHERE id NOT IN (
                SELECT DISTINCT student_id
                FROM attendance
                WHERE DATE(check_in_time) = ?
            )
        ''', (today,))
        
        absent = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return present, absent
    
    def get_session_attendance_records(self, session_id: int) -> List[Dict]:
        """某场签到的签到明细（按签到时间倒序）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT a.id, a.student_id, s.name, a.check_in_time, a.confidence
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.session_id = ?
            ORDER BY a.check_in_time DESC
            ''',
            (session_id,),
        )
        results = cursor.fetchall()
        conn.close()
        records = []
        for result in results:
            records.append(
                {
                    "id": result[0],
                    "student_id": result[1],
                    "name": result[2],
                    "check_in_time": result[3],
                    "confidence": result[4],
                }
            )
        return records
    
    def get_session_summary(
        self, session_id: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        某场：已签到列表、未到列表（全班减去本场已签到）
        """
        present_records = self.get_session_attendance_records(session_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT id, name FROM students
            WHERE id NOT IN (
                SELECT DISTINCT student_id FROM attendance WHERE session_id = ?
            )
            ORDER BY id
            ''',
            (session_id,),
        )
        absent = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
        conn.close()
        
        return present_records, absent
    
    def list_attendance_sessions(self, limit: int = 30) -> List[Dict]:
        """最近若干场签到（用于历史下拉）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT s.id, s.started_at, s.ended_at,
                   (SELECT COUNT(DISTINCT a.student_id)
                    FROM attendance a WHERE a.session_id = s.id) AS present_n
            FROM attendance_sessions s
            ORDER BY s.id DESC
            LIMIT ?
            ''',
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]
        conn.close()
        
        sessions = []
        for row in rows:
            sid, started, ended, present_n = row[0], row[1], row[2], row[3] or 0
            absent_n = max(0, total_students - present_n)
            label = f"#{sid} {started}"
            if ended:
                label += f" ~ {ended}"
            else:
                label += " (进行中)"
            sessions.append(
                {
                    "id": sid,
                    "started_at": started,
                    "ended_at": ended,
                    "present_count": present_n,
                    "absent_count": absent_n,
                    "label": label,
                }
            )
        return sessions
    
    def get_latest_session_id(self) -> Optional[int]:
        """id 最大的一场（通常即最近一场）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM attendance_sessions")
        row = cursor.fetchone()
        conn.close()
        if row and row[0] is not None:
            return int(row[0])
        return None
    
    def get_session_row(self, session_id: int) -> Optional[Dict]:
        """单场签到元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT id, started_at, ended_at FROM attendance_sessions WHERE id = ?
            ''',
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {"id": row[0], "started_at": row[1], "ended_at": row[2]}
    
    def add_behavior_evidence(
        self,
        behavior_type: str,
        track_id: int,
        image_path: str,
        confidence: float,
        trigger_reason: str,
        detail_json: Optional[str] = None,
    ) -> Optional[int]:
        """写入一条异常行为抓拍记录。"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # 使用本地时间
            local_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                '''
                INSERT INTO behavior_evidence
                (behavior_type, track_id, image_path, confidence, trigger_reason, detail_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (behavior_type, track_id, image_path, confidence, trigger_reason, detail_json, local_now),
            )
            rid = cursor.lastrowid
            conn.commit()
            conn.close()
            return int(rid) if rid is not None else None
        except Exception as e:
            print(f"写入 behavior_evidence 失败: {e}")
            return None
    
    def list_behavior_evidence(self, limit: int = 100) -> List[Dict]:
        """按时间倒序列出抓拍记录。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT id, behavior_type, track_id, image_path, confidence,
                   trigger_reason, detail_json, created_at
            FROM behavior_evidence
            ORDER BY id DESC
            LIMIT ?
            ''',
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "behavior_type": r[1],
                    "track_id": r[2],
                    "image_path": r[3],
                    "confidence": r[4],
                    "trigger_reason": r[5],
                    "detail_json": r[6],
                    "created_at": r[7],
                }
            )
        return out
    
    def list_behavior_evidence_for_date(self, d: Optional[date] = None, limit: int = 500) -> List[Dict]:
        """指定日期的异常抓拍记录（默认今天），按时间升序。"""
        if d is None:
            d = date.today()
        day_str = d.isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''
                SELECT id, behavior_type, track_id, image_path, confidence,
                       trigger_reason, detail_json, created_at
                FROM behavior_evidence
                WHERE date(created_at) = ?
                ORDER BY id ASC
                LIMIT ?
                ''',
                (day_str, limit),
            )
        except sqlite3.OperationalError:
            conn.close()
            return []
        rows = cursor.fetchall()
        conn.close()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "behavior_type": r[1],
                    "track_id": r[2],
                    "image_path": r[3],
                    "confidence": r[4],
                    "trigger_reason": r[5],
                    "detail_json": r[6],
                    "created_at": r[7],
                }
            )
        return out
    
    def clear_old_attendance(self, days: int = 30):
        """
        清理旧考勤记录
        Args:
            days: 保留天数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            DELETE FROM attendance
            WHERE check_in_time < DATE('now', '-{} days')
            '''.format(days)
        )
        cursor.execute(
            '''
            DELETE FROM attendance_sessions
            WHERE started_at < DATE('now', '-{} days')
            '''.format(days)
        )
        
        conn.commit()
        conn.close()
