import numpy as np
import time
from typing import Dict, List, Optional
from .connection import db_connection
from ..core.logger import get_logger

logger = get_logger(__name__)


class VoiceprintDB:
    """声纹数据库操作类，负责声纹特征的存储与读取"""

    def save_voiceprint(self, speaker_id: str, emb: np.ndarray) -> bool:
        """
        保存或更新声纹特征

        Args:
            speaker_id: 说话人ID
            emb: 声纹特征向量

        Returns:
            bool: 操作是否成功
        """
        try:
            with db_connection.get_cursor() as cursor:
                sql = """
                INSERT INTO voiceprints (speaker_id, feature_vector)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE feature_vector=VALUES(feature_vector)
                """
                cursor.execute(sql, (speaker_id, emb.tobytes()))
                logger.success(f"声纹特征保存成功: {speaker_id}")
                return True
        except Exception as e:
            logger.fail(f"保存声纹特征失败 {speaker_id}: {e}")
            return False

    def get_voiceprints(
        self, speaker_ids: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        获取指定说话人ID的声纹特征（如未指定则获取全部）

        Args:
            speaker_ids: 说话人ID列表

        Returns:
            Dict[str, np.ndarray]: {speaker_id: 特征向量}
        """
        start_time = time.time()
        query_type = (
            f"指定ID查询({len(speaker_ids) if speaker_ids else 0}个)"
            if speaker_ids
            else "全量查询"
        )
        logger.info(f"开始数据库查询: {query_type}")

        try:
            with db_connection.get_cursor() as cursor:
                if speaker_ids:
                    format_strings = ",".join(["%s"] * len(speaker_ids))
                    sql = f"SELECT speaker_id, feature_vector FROM voiceprints WHERE speaker_id IN ({format_strings})"
                    cursor.execute(sql, tuple(speaker_ids))
                else:
                    sql = "SELECT speaker_id, feature_vector FROM voiceprints"
                    cursor.execute(sql)

                fetch_start = time.time()
                results = cursor.fetchall()
                fetch_time = time.time() - fetch_start
                logger.info(
                    f"数据库查询完成，获取到{len(results)}条记录，查询耗时: {fetch_time:.3f}秒"
                )

                # 将数据库中的二进制特征转为numpy数组
                convert_start = time.time()
                voiceprints = {
                    row[0]: np.frombuffer(row[1], dtype=np.float32) for row in results
                }
                convert_time = time.time() - convert_start
                logger.info(f"数据转换完成，转换耗时: {convert_time:.3f}秒")

                total_time = time.time() - start_time
                logger.info(
                    f"获取到 {len(voiceprints)} 个声纹特征，总耗时: {total_time:.3f}秒"
                )
                return voiceprints
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"获取声纹特征失败，总耗时: {total_time:.3f}秒，错误: {e}")
            return {}

    def delete_voiceprint(self, speaker_id: str) -> bool:
        """
        删除指定说话人的声纹特征

        Args:
            speaker_id: 说话人ID

        Returns:
            bool: 操作是否成功
        """
        try:
            with db_connection.get_cursor() as cursor:
                sql = "DELETE FROM voiceprints WHERE speaker_id = %s"
                cursor.execute(sql, (speaker_id,))
                if cursor.rowcount > 0:
                    logger.info(f"声纹特征删除成功: {speaker_id}")
                    return True
                else:
                    logger.warning(f"未找到要删除的声纹特征: {speaker_id}")
                    return False
        except Exception as e:
            logger.error(f"删除声纹特征失败 {speaker_id}: {e}")
            return False

    def count_voiceprints(self) -> int:
        """
        获取声纹特征总数

        Returns:
            int: 声纹特征总数
        """
        start_time = time.time()
        logger.info("开始查询声纹特征总数...")

        try:
            with db_connection.get_cursor() as cursor:
                sql = "SELECT COUNT(*) FROM voiceprints"
                cursor.execute(sql)
                result = cursor.fetchone()
                count = result[0] if result else 0

                total_time = time.time() - start_time
                logger.info(f"声纹特征总数查询完成: {count}，耗时: {total_time:.3f}秒")
                return count
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"获取声纹特征总数失败，总耗时: {total_time:.3f}秒，错误: {e}")
            return 0

    def get_voiceprint_list(self, page: int = 1, page_size: int = 10) -> Dict:
        """
        获取声纹列表，支持分页

        Args:
            page: 页码（从1开始）
            page_size: 每页数量

        Returns:
            Dict: 包含总条数、列表数据的字典
        """
        start_time = time.time()
        logger.info(f"开始查询声纹列表，页码: {page}，每页数量: {page_size}")

        try:
            # 计算偏移量
            offset = (page - 1) * page_size

            with db_connection.get_cursor() as cursor:
                # 查询总数
                cursor.execute("SELECT COUNT(*) FROM voiceprints")
                total_result = cursor.fetchone()
                total = total_result[0] if total_result else 0

                # 查询数据
                sql = """
                SELECT id, speaker_id, created_at, updated_at
                FROM voiceprints
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (page_size, offset))
                results = cursor.fetchall()

                # 构造返回数据
                voiceprint_list = []
                for row in results:
                    voiceprint_list.append({
                        "id": row[0],
                        "speaker_id": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "updated_at": row[3].isoformat() if row[3] else None
                    })

                total_time = time.time() - start_time
                logger.info(f"声纹列表查询完成，总数: {total}，当前页: {len(voiceprint_list)}，耗时: {total_time:.3f}秒")

                return {
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "list": voiceprint_list
                }
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"获取声纹列表失败，总耗时: {total_time:.3f}秒，错误: {e}")
            return {
                "total": 0,
                "page": page,
                "page_size": page_size,
                "list": []
            }


# 全局声纹数据库操作实例
voiceprint_db = VoiceprintDB()
