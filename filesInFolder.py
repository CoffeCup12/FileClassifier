from pathlib import Path
import docx2txt

class Extractor:
    def __init__(self, path):
        self.path = path

    def showPath(self):
        print(f"Path: {self.path}")

    def pdfsInFolder(self):
        folder = Path(self.path)
        return [str(file.resolve()) for file in folder.rglob("*.pdf")]
    
    def docxsInFolder(self):
        folder = Path(self.path)
        return [str(file.resolve()) for file in folder.rglob("*.docx")]
    
    def docxscontents(self):
        pathes = self.docxsInFolder()
        result = []
        for i in range(len(pathes)):
            content = docx2txt.process(pathes[i])
            result.append([pathes[i], content])

        return result





if __name__ == "__main__":
    target_folder = input("Enter the folder: ").strip()
    extractor1 = Extractor(target_folder)
    print("PDF:")
    print(extractor1.pdfsInFolder())
    print("\nDOCX:")
    print(extractor1.docxsInFolder())
    print("\nDOCX_content:")
    print(extractor1.docxscontents())