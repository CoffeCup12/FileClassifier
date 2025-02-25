from app import app
import subprocess

subprocess.run(["make", "run"])

if __name__ == "__main__":
    app.run(debug=True)