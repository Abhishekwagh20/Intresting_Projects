# memory_manager.py
import os
import hashlib
from pathlib import Path
from vector_store import VectorStore  # Import the new VectorStore

class ProjectMemory:
    def __init__(self):
        self.project_structure = {}
        self.file_contents = {}
        self.dependency_graph = {}
        self.vector_store = VectorStore()  # Initialize vector store

    def load_project(self, root_path):
        """Recursively load project files"""
        root = Path(root_path)
        for file_path in root.glob('**/*'):
            if file_path.is_file() and not self._is_ignored(file_path):
                self._store_file(file_path)
                
        self._build_dependency_graph()
        self.vector_store.add_documents(self.file_contents)  # Add files to vector store

    def _store_file(self, file_path):
        """Store file content with versioning"""
        content = file_path.read_text(encoding='utf-8')
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        self.project_structure[str(file_path)] = {
            'size': os.path.getsize(file_path),
            'modified': os.path.getmtime(file_path),
            'hash': file_hash,
            'dependencies': []
        }
        
        self.file_contents[str(file_path)] = content
        
    def _build_dependency_graph(self):
        """Analyze file dependencies (basic implementation)"""
        for file_path, content in self.file_contents.items():
            self.project_structure[file_path]['dependencies'] = self._find_dependencies(content)
            
    def _find_dependencies(self, content):
        """Detect imports/requires/references (customize per your stack)"""
        # Example: Python imports
        imports = []
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                imports.append(line.strip())
        return imports
        
    def get_relevant_context(self, query):
        """Retrieve related code context for a query using vector search"""
        results = self.vector_store.search(query)
        context = []
        for result in results:
            context.append(f"File: {result['file_path']}\nScore: {result['score']:.2f}\n{result['content']}")
        return "\n\n".join(context)

    def _is_ignored(self, file_path):
        # Add custom ignore patterns
        ignore = ['.git', 'venv', '__pycache__', '.env']
        return any(pattern in str(file_path) for pattern in ignore)