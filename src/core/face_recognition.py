"""
人脸识别模块
优先使用 dlib（需安装且具备模型文件）；否则使用 facenet-pytorch（MTCNN + InceptionResnetV1），无需 dlib。
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from scipy.spatial.distance import cosine

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PRETRAINED_DIR,
    DLIB_FACE_MODEL,
    DLIB_LANDMARK_MODEL,
    FACE_CONFIG,
    DEVICE,
)


def _use_dlib_backend() -> bool:
    """同时满足：已安装 dlib，且两个 dlib 模型文件均存在。"""
    try:
        import dlib  # noqa: F401
    except ImportError:
        return False
    face_path = os.path.join(PRETRAINED_DIR, DLIB_FACE_MODEL)
    land_path = os.path.join(PRETRAINED_DIR, DLIB_LANDMARK_MODEL)
    return os.path.exists(face_path) and os.path.exists(land_path)


class _FaceRecognizerDlib:
    """dlib 128 维特征（与历史数据兼容）。"""

    embedding_dim = 128

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        landmark_model_path: Optional[str] = None,
        match_threshold: float = FACE_CONFIG["face_match_threshold"],
    ):
        import dlib

        self.match_threshold = match_threshold

        if face_model_path is None:
            face_model_path = os.path.join(PRETRAINED_DIR, DLIB_FACE_MODEL)
        if landmark_model_path is None:
            landmark_model_path = os.path.join(PRETRAINED_DIR, DLIB_LANDMARK_MODEL)

        self.detector = dlib.get_frontal_face_detector()

        if os.path.exists(landmark_model_path):
            self.shape_predictor = dlib.shape_predictor(landmark_model_path)
        else:
            print(f"警告: 关键点模型不存在: {landmark_model_path}")
            self.shape_predictor = None

        if os.path.exists(face_model_path):
            self.face_recognizer = dlib.face_recognition_model_v1(face_model_path)
        else:
            print(f"警告: 人脸识别模型不存在: {face_model_path}")
            self.face_recognizer = None

    def detect_faces(
        self, frame: np.ndarray, upsample: int = 1
    ) -> List[Tuple[int, int, int, int]]:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        detections = self.detector(rgb_frame, upsample)
        faces = []
        for det in detections:
            faces.append((det.left(), det.top(), det.right(), det.bottom()))
        return faces

    def get_face_encoding(
        self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        if self.shape_predictor is None or self.face_recognizer is None:
            print("模型未加载，无法提取特征")
            return None

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        import dlib

        left, top, right, bottom = face_rect
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        shape = self.shape_predictor(rgb_frame, dlib_rect)
        face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_frame, shape)
        return np.array(face_descriptor)

    def get_face_landmarks(
        self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        if self.shape_predictor is None:
            return None

        import dlib

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        left, top, right, bottom = face_rect
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        shape = self.shape_predictor(rgb_frame, dlib_rect)
        return np.array([[p.x, p.y] for p in shape.parts()])

    def compare_faces(
        self,
        face_encoding1: np.ndarray,
        face_encoding2: np.ndarray,
        metric: str = "euclidean",
    ) -> float:
        if metric == "euclidean":
            return float(np.linalg.norm(face_encoding1 - face_encoding2))
        if metric == "cosine":
            return float(cosine(face_encoding1, face_encoding2))
        raise ValueError(f"不支持的距离度量: {metric}")

    def find_best_match(
        self,
        face_encoding: np.ndarray,
        known_encodings: List[np.ndarray],
        known_names: List[str],
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        if threshold is None:
            threshold = self.match_threshold
        if len(known_encodings) == 0:
            return None, 1.0

        distances = []
        for encoding in known_encodings:
            distances.append(self.compare_faces(face_encoding, encoding))

        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]
        if min_dist < threshold:
            return known_names[min_idx], min_dist
        return None, min_dist

    def recognize_face(
        self,
        frame: np.ndarray,
        face_rect: Optional[Tuple[int, int, int, int]] = None,
        known_encodings: Optional[List[np.ndarray]] = None,
        known_names: Optional[List[str]] = None,
    ) -> Dict:
        result: Dict[str, Any] = {
            "success": False,
            "face_rect": None,
            "name": None,
            "confidence": 0.0,
            "encoding": None,
        }

        if face_rect is None:
            faces = self.detect_faces(frame)
            if len(faces) == 0:
                return result
            face_rect = faces[0]

        result["face_rect"] = face_rect
        encoding = self.get_face_encoding(frame, face_rect)
        if encoding is None:
            return result

        result["encoding"] = encoding
        result["success"] = True

        if known_encodings is not None and known_names is not None:
            name, dist = self.find_best_match(encoding, known_encodings, known_names)
            result["name"] = name
            result["confidence"] = max(0.0, min(1.0, 1.0 - float(dist)))

        return result

    def align_face(
        self,
        frame: np.ndarray,
        face_rect: Tuple[int, int, int, int],
        size: int = 256,
    ) -> Optional[np.ndarray]:
        if self.shape_predictor is None:
            return None

        landmarks = self.get_face_landmarks(frame, face_rect)
        if landmarks is None:
            return None

        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        left, top, right, bottom = face_rect
        face_crop = aligned[top:bottom, left:right]
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (size, size))
        return face_crop


class _FaceRecognizerTorch:
    """facenet-pytorch：512 维归一化特征，不依赖 dlib。"""

    embedding_dim = 512

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        landmark_model_path: Optional[str] = None,
        match_threshold: Optional[float] = None,
    ):
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError as e:
            raise ImportError(
                "未安装 facenet-pytorch，无法使用免 dlib 人脸方案。请执行: pip install facenet-pytorch"
            ) from e

        if match_threshold is None:
            match_threshold = FACE_CONFIG.get(
                "face_match_threshold_torch", 0.95
            )
        self.match_threshold = float(match_threshold)

        self._torch = torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # 与 DEVICE 字符串一致时优先用 GPU
        if "cuda" in str(DEVICE).lower() and torch.cuda.is_available():
            self.device = torch.device(DEVICE if ":" in DEVICE else "cuda:0")

        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            device=self.device,
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(
            self.device
        )

    def detect_faces(
        self, frame: np.ndarray, upsample: int = 1
    ) -> List[Tuple[int, int, int, int]]:
        del upsample
        pil = self._bgr_to_pil(frame)
        boxes, _ = self.mtcnn.detect(pil)
        if boxes is None:
            return []
        faces = []
        for b in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            faces.append((x1, y1, x2, y2))
        return faces

    def _bgr_to_pil(self, frame: np.ndarray):
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def get_face_encoding(
        self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        left, top, right, bottom = face_rect
        h, w = frame.shape[:2]
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        if right <= left or bottom <= top:
            return None

        crop = frame[top:bottom, left:right]
        # 检查裁剪区域是否有效
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        pil = self._bgr_to_pil(crop)

        # 使用 try-except 捕获 MTCNN 可能的异常
        try:
            img_tensor = self.mtcnn(pil)
        except Exception as e:
            # MTCNN 内部错误（如未检测到人脸时 torch.cat 空列表）
            return None

        if img_tensor is None:
            return None

        torch = self._torch
        with torch.no_grad():
            emb = self.resnet(img_tensor.unsqueeze(0).to(self.device))
        vec = emb.squeeze(0).cpu().numpy().astype(np.float64)
        n = np.linalg.norm(vec)
        if n > 1e-6:
            vec = vec / n
        return vec

    def get_face_landmarks(
        self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        del frame, face_rect
        return None

    def compare_faces(
        self,
        face_encoding1: np.ndarray,
        face_encoding2: np.ndarray,
        metric: str = "euclidean",
    ) -> float:
        if metric == "euclidean":
            return float(np.linalg.norm(face_encoding1 - face_encoding2))
        if metric == "cosine":
            return float(cosine(face_encoding1, face_encoding2))
        raise ValueError(f"不支持的距离度量: {metric}")

    def find_best_match(
        self,
        face_encoding: np.ndarray,
        known_encodings: List[np.ndarray],
        known_names: List[str],
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        if threshold is None:
            threshold = self.match_threshold
        if len(known_encodings) == 0:
            return None, 1.0

        distances = []
        for enc in known_encodings:
            if enc.shape[0] != face_encoding.shape[0]:
                distances.append(1e9)
            else:
                distances.append(self.compare_faces(face_encoding, enc))

        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]
        if min_dist < threshold:
            return known_names[min_idx], min_dist
        return None, min_dist

    def recognize_face(
        self,
        frame: np.ndarray,
        face_rect: Optional[Tuple[int, int, int, int]] = None,
        known_encodings: Optional[List[np.ndarray]] = None,
        known_names: Optional[List[str]] = None,
    ) -> Dict:
        result: Dict[str, Any] = {
            "success": False,
            "face_rect": None,
            "name": None,
            "confidence": 0.0,
            "encoding": None,
        }

        if face_rect is None:
            faces = self.detect_faces(frame)
            if len(faces) == 0:
                return result
            face_rect = max(
                faces,
                key=lambda r: (r[2] - r[0]) * (r[3] - r[1]),
            )

        result["face_rect"] = face_rect
        encoding = self.get_face_encoding(frame, face_rect)
        if encoding is None:
            return result

        result["encoding"] = encoding
        result["success"] = True

        if known_encodings is not None and known_names is not None:
            name, dist = self.find_best_match(encoding, known_encodings, known_names)
            result["name"] = name
            result["confidence"] = max(0.0, min(1.0, 1.0 - float(dist)))

        return result

    def align_face(
        self,
        frame: np.ndarray,
        face_rect: Tuple[int, int, int, int],
        size: int = 256,
    ) -> Optional[np.ndarray]:
        left, top, right, bottom = face_rect
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (size, size))


class FaceRecognizer:
    """
    统一入口：自动选择 dlib 或 PyTorch 方案。
    数据库中若存在与当前方案维度不一致的旧特征，请在加载时过滤（见考勤线程）。
    """

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        landmark_model_path: Optional[str] = None,
        match_threshold: float = FACE_CONFIG["face_match_threshold"],
    ):
        if _use_dlib_backend():
            self._impl: Any = _FaceRecognizerDlib(
                face_model_path, landmark_model_path, match_threshold
            )
            self.backend = "dlib"
        else:
            torch_thr = FACE_CONFIG.get("face_match_threshold_torch", 0.95)
            self._impl = _FaceRecognizerTorch(
                face_model_path, landmark_model_path, torch_thr
            )
            self.backend = "facenet_pytorch"

        self.embedding_dim = self._impl.embedding_dim
        self.match_threshold = getattr(self._impl, "match_threshold", match_threshold)

    def detect_faces(
        self, frame: np.ndarray, upsample: int = 1
    ) -> List[Tuple[int, int, int, int]]:
        return self._impl.detect_faces(frame, upsample)

    def get_face_encoding(
        self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        return self._impl.get_face_encoding(frame, face_rect)

    def get_face_landmarks(
        self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        return self._impl.get_face_landmarks(frame, face_rect)

    def compare_faces(
        self,
        face_encoding1: np.ndarray,
        face_encoding2: np.ndarray,
        metric: str = "euclidean",
    ) -> float:
        return self._impl.compare_faces(face_encoding1, face_encoding2, metric)

    def find_best_match(
        self,
        face_encoding: np.ndarray,
        known_encodings: List[np.ndarray],
        known_names: List[str],
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        return self._impl.find_best_match(
            face_encoding, known_encodings, known_names, threshold
        )

    def recognize_face(
        self,
        frame: np.ndarray,
        face_rect: Optional[Tuple[int, int, int, int]] = None,
        known_encodings: Optional[List[np.ndarray]] = None,
        known_names: Optional[List[str]] = None,
    ) -> Dict:
        return self._impl.recognize_face(
            frame, face_rect, known_encodings, known_names
        )

    def align_face(
        self,
        frame: np.ndarray,
        face_rect: Tuple[int, int, int, int],
        size: int = 256,
    ) -> Optional[np.ndarray]:
        return self._impl.align_face(frame, face_rect, size)


class FaceRecognitionPipeline:
    """人脸识别流水线"""

    def __init__(
        self,
        face_model_path: Optional[str] = None,
        landmark_model_path: Optional[str] = None,
        liveness_model_path: Optional[str] = None,
    ):
        self.recognizer = FaceRecognizer(face_model_path, landmark_model_path)

        self.liveness_detector = None
        if liveness_model_path is not None:
            try:
                from .liveness_detector import LivenessDetector

                self.liveness_detector = LivenessDetector(liveness_model_path)
            except Exception as e:
                print(f"活体检测模型加载失败: {e}")

        self.known_encodings: List[np.ndarray] = []
        self.known_ids: List[str] = []

    def register_face(
        self,
        frame: np.ndarray,
        student_id: str,
        check_liveness: bool = True,
    ) -> Dict:
        result: Dict[str, Any] = {
            "success": False,
            "message": "",
            "is_live": None,
            "face_rect": None,
            "encoding": None,
        }

        faces = self.recognizer.detect_faces(frame)
        if len(faces) == 0:
            result["message"] = "未检测到人脸"
            return result

        face_rect = max(
            faces,
            key=lambda r: (r[2] - r[0]) * (r[3] - r[1]),
        )
        result["face_rect"] = face_rect

        if check_liveness and self.liveness_detector is not None:
            is_live, score = self.liveness_detector.detect(frame, face_rect)
            result["is_live"] = is_live
            result["liveness_score"] = score
            if not is_live:
                result["message"] = "活体检测失败，请使用真实人脸"
                return result

        encoding = self.recognizer.get_face_encoding(frame, face_rect)
        if encoding is None:
            result["message"] = "特征提取失败"
            return result

        result["encoding"] = encoding
        result["success"] = True
        result["message"] = "注册成功"

        self.known_encodings.append(encoding)
        self.known_ids.append(student_id)

        return result

    def identify(self, frame: np.ndarray) -> List[Dict]:
        results = []
        faces = self.recognizer.detect_faces(frame)

        for face_rect in faces:
            result = self.recognizer.recognize_face(
                frame,
                face_rect,
                self.known_encodings,
                self.known_ids,
            )

            if self.liveness_detector is not None:
                is_live, score = self.liveness_detector.detect(frame, face_rect)
                result["is_live"] = is_live
                result["liveness_score"] = score

            results.append(result)

        return results

    def load_known_faces(self, encodings: List[np.ndarray], ids: List[str]):
        self.known_encodings = encodings
        self.known_ids = ids

    def clear_known_faces(self):
        self.known_encodings.clear()
        self.known_ids.clear()


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    print(f"人脸后端: {recognizer.backend}, 特征维度: {recognizer.embedding_dim}")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    faces = recognizer.detect_faces(test_image)
    print(f"检测到 {len(faces)} 个人脸")
