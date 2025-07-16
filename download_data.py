#!/usr/bin/env python3
"""
数据文件下载脚本
用于下载项目所需的大文件
"""

import os
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

# 文件下载配置
FILE_CONFIGS = {
    # 数据集文件
    "Data/meld_data.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 88.28 * 1024 * 1024,  # MB to bytes
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_data_act.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 71.55 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_data_original.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 71.04 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_train.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 279.26 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_test.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 75.33 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_context_train.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 1689.28 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_context_test.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 441.39 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/meld_context_valid.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 187.55 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/iemocap_context_train.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 248.94 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/iemocap_context_test.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 94.00 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "Data/iemocap_context_valid.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 87.57 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },

    # 模型文件
    "feature_extract/BEATs_iter3_plus_AS2M.pt": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 344.75 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },

    # 特征文件
    "feature_extract/Features4Quantum/fudan_train/features.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 109.50 * 1024 * 1024,
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "feature_extract/Features4Quantum/fudan_test/features.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 0,  # 需要填写实际大小
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "feature_extract/Features4Quantum/fudan_val/features.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 0,  # 需要填写实际大小
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "feature_extract/Features4Quantum/chinese_train/features.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 0,  # 需要填写实际大小
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "feature_extract/Features4Quantum/chinese_test/features.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 0,  # 需要填写实际大小
        "md5": "YOUR_MD5_HASH_HERE"
    },
    "feature_extract/Features4Quantum/chinese_val/features.pkl": {
        "url": "YOUR_DOWNLOAD_URL_HERE",
        "size": 0,  # 需要填写实际大小
        "md5": "YOUR_MD5_HASH_HERE"
    }
}


def calculate_md5(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, file_path, expected_size=None, expected_md5=None):
    """下载文件"""
    # 创建目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 检查文件是否已存在
    if os.path.exists(file_path):
        print(f"文件 {file_path} 已存在，跳过下载")
        if expected_md5:
            actual_md5 = calculate_md5(file_path)
            if actual_md5 == expected_md5:
                print(f"✓ MD5验证通过")
                return True
            else:
                print(f"✗ MD5验证失败，重新下载")
                os.remove(file_path)

    try:
        print(f"正在下载 {file_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        if expected_size and total_size != expected_size:
            print(f"警告: 文件大小不匹配 (期望: {expected_size}, 实际: {total_size})")

        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(file_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # 验证MD5
        if expected_md5:
            actual_md5 = calculate_md5(file_path)
            if actual_md5 == expected_md5:
                print(f"✓ 下载完成，MD5验证通过")
                return True
            else:
                print(f"✗ MD5验证失败")
                os.remove(file_path)
                return False
        else:
            print(f"✓ 下载完成")
            return True

    except Exception as e:
        print(f"下载失败: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False


def main():
    """主函数"""
    print("开始下载数据文件...")
    print("=" * 50)

    success_count = 0
    total_count = len(FILE_CONFIGS)

    for file_path, config in FILE_CONFIGS.items():
        url = config["url"]
        if url == "YOUR_DOWNLOAD_URL_HERE":
            print(f"跳过 {file_path} (URL未配置)")
            continue

        if download_file(url, file_path, config.get("size"), config.get("md5")):
            success_count += 1
        print("-" * 30)

    print("=" * 50)
    print(f"下载完成: {success_count}/{total_count} 个文件")

    if success_count < total_count:
        print("\n注意: 部分文件下载失败或未配置URL")
        print("请检查 DATA_DOWNLOAD.md 文件获取手动下载链接")


if __name__ == "__main__":
    main()
