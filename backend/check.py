# 

import importlib

libraries = {
    "OpenCV (cv2)": "cv2",
    "NumPy": "numpy",
    "Pillow": "PIL",
    "Scikit-learn": "sklearn",
    "SciPy": "scipy",
    "PyTorch": "torch",
    "Torchvision": "torchvision",
    "Matplotlib": "matplotlib",
    "TQDM": "tqdm",
    "YAML": "yaml",
    "Requests": "requests",
    "Flask": "flask",
    "Flask-CORS": "flask_cors",
}

print("\n========== DEPENDENCY CHECK ==========\n")

for name, module in libraries.items():
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "Installed")
        print(f"[OK]   {name}: {version}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")

print("\n========== CHECK COMPLETE ==========")