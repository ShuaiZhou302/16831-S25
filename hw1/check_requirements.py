import pkg_resources
import sys

requirements_path = 'hw1/requirements.txt'
missing = []

with open(requirements_path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            pkg_resources.require(line)
            print(f"[OK] {line}")
        except Exception as e:
            print(f"[MISSING/ERROR] {line} -> {e}")
            missing.append(line)

if missing:
    print("\n以下依赖未正确安装或版本不符：")
    for m in missing:
        print(m)
else:
    print("\n所有 requirements.txt 依赖均已正确安装！")

# 额外检查 free-mujoco-py 是否能 import
try:
    import free_mujoco_py
    print("[IMPORT OK] free-mujoco-py 可以正常 import")
except ImportError as e:
    print(f"[IMPORT ERROR] free-mujoco-py 无法 import: {e}")

if missing:
    sys.exit(1)
