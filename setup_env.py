import os
import sys
import subprocess
from pathlib import Path

VENV_DIR = ".venv"
REQ_FILE = "requirements.txt"

def main():
    print("📦 Starting environment setup...")
    
    # 1. Create Virtual Environment
    if not os.path.exists(VENV_DIR):
        print(f"Creating isolated virtual environment in '{VENV_DIR}'...")
        # Using the standard library 'venv' to avoid external dependencies
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    else:
        print(f"Virtual environment '{VENV_DIR}' already exists.")
    
    # 2. Determine the correct path to the venv's Python executable
    if sys.platform == "win32":
        venv_python = Path(VENV_DIR) / "Scripts" / "python.exe"
        activate_cmd = f"{VENV_DIR}\\Scripts\\activate"
    else:
        venv_python = Path(VENV_DIR) / "bin" / "python"
        activate_cmd = f"source {VENV_DIR}/bin/activate"
    
    if not venv_python.exists():
        sys.exit(f"❌ Error: Python executable not found at {venv_python}")

    # 3. Upgrade pip (prevents annoying warning messages)
    print("🔄 Upgrading pip...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)
    
    # 4. Install dependencies safely inside the venv
    if os.path.exists(REQ_FILE):
        print(f"📥 Installing dependencies from {REQ_FILE}...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", REQ_FILE], check=True)
    else:
        print(f"⚠️ {REQ_FILE} not found. Creating an empty environment.")
    
    # 5. Hand off instructions to the user
    print("\n✅ Setup complete! Everything is installed in the isolated environment.")
    print("To activate it and start working, run the following command:\n")
    print("-" * 50)
    print(activate_cmd)
    print("-" * 50)

if __name__ == "__main__":
    main()
