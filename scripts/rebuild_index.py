"""
重建全量索引脚本
用法：python scripts/rebuild_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import rebuild_all

if __name__ == "__main__":
    print("🔄 开始全量重建塔罗知识库索引...")
    result = rebuild_all(verbose=True)
    print(f"\n✅ 重建完成！共处理 {result['files']} 个文件，{result['chunks']} 个 chunk")
