from pathlib import Path

def filesInFolder(folder):
    folder = Path(folder)
    return [str(file.resolve()) for file in folder.rglob("*.pdf")]


if __name__ == "__main__":
    target_folder = input("Enter the folder: ").strip()
    
    print("Result(s): ")
    print(*filesInFolder(target_folder), sep="\n")