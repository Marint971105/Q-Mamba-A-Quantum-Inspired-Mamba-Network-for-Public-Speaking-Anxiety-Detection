#!/usr/bin/env python3
"""
数据文件验证脚本
用于验证下载的数据文件是否完整
"""

import os
import hashlib
from pathlib import Path

# 文件验证配置
FILE_CONFIGS = {
    "Data/meld_data.pkl": {
        "size": 88.28 * 1024 * 1024,  # MB to bytes
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_data_act.pkl": {
        "size": 71.55 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_data_original.pkl": {
        "size": 71.04 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_train.pkl": {
        "size": 279.26 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_test.pkl": {
        "size": 75.33 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_context_train.pkl": {
        "size": 1689.28 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_context_test.pkl": {
        "size": 441.39 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_context_valid.pkl": {
        "size": 187.55 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/iemocap_context_train.pkl": {
        "size": 248.94 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/iemocap_context_test.pkl": {
        "size": 94.00 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/iemocap_context_valid.pkl": {
        "size": 87.57 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "feature_extract/BEATs_iter3_plus_AS2M.pt": {
        "size": 344.75 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "feature_extract/Features4Quantum/fudan_train/features.pkl": {
        "size": 109.50 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    }
}


def calculate_md5(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"计算MD5时出错: {e}")
        return None


def get_file_size(file_path):
    """获取文件大小"""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        print(f"获取文件大小时出错: {e}")
        return None


def verify_file(file_path, expected_size=None, expected_md5=None):
    """验证单个文件"""
    print(f"验证文件: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"✗ 文件不存在")
        return False

    # 检查文件大小
    actual_size = get_file_size(file_path)
    if actual_size is None:
        return False

    if expected_size:
        if actual_size != expected_size:
            print(f"✗ 文件大小不匹配 (期望: {expected_size}, 实际: {actual_size})")
            return False
        else:
            print(f"✓ 文件大小正确: {actual_size / (1024*1024):.2f} MB")

    # 检查MD5
    if expected_md5 and expected_md5 != "YOUR_MD5_HASH_HERE":
        actual_md5 = calculate_md5(file_path)
        if actual_md5 is None:
            return False

        if actual_md5 == expected_md5:
            print(f"✓ MD5验证通过")
        else:
            print(f"✗ MD5验证失败 (期望: {expected_md5}, 实际: {actual_md5})")
            return False
    else:
        print(f"⚠ MD5未配置，跳过验证")

    print(f"✓ 文件验证通过")
    return True


def main():
    """主函数"""
    print("开始验证数据文件...")
    print("=" * 50)

    success_count = 0
    total_count = len(FILE_CONFIGS)

    for file_path, config in FILE_CONFIGS.items():
        if verify_file(file_path, config.get("size"), config.get("md5")):
            success_count += 1
        print("-" * 30)

    print("=" * 50)
    print(f"验证完成: {success_count}/{total_count} 个文件通过验证")

    if success_count == total_count:
        print("🎉 所有文件验证通过！")
    else:
        print(f"⚠ {total_count - success_count} 个文件验证失败")
        print("请检查文件是否完整下载")


if __name__ == "__main__":
    main()
