import sys
import os
import io
import traceback
import ast
import importlib
import builtins
import types
from collections import defaultdict
import subprocess
from typing import Dict, Set
class CustomImporter:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def find_spec(self, fullname, path, target=None):
        if fullname in self.interpreter.modules:
            return importlib.util.spec_from_loader(fullname, loader=None)

class RobustCodeInterpreter:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.globals = {'__builtins__': builtins}
        self.modules = {}
        self.module_code = {}
        self.dependencies = defaultdict(set)
        self.importer = CustomImporter(self)
        sys.meta_path.insert(0, self.importer)
        self._prepared= False
        self.local_files = set()
        self.python_executable = sys.executable

    def add_module(self, filename, code):
        module_name = os.path.splitext(filename)[0]
        self.module_code[module_name] = code

    def _update_dependencies(self):
        for module_name, code in self.module_code.items():
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.dependencies[module_name].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:  # relative import
                        parts = module_name.split('.')
                        prefix = '.'.join(parts[:-node.level])
                        if node.module:
                            self.dependencies[module_name].add(f"{prefix}.{node.module}")
                        else:
                            self.dependencies[module_name].add(prefix)
                    else:
                        self.dependencies[module_name].add(node.module)

    def _load_local_files(self):
        for root, dirs, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith('.py'):
                    module_path = os.path.relpath(os.path.join(root, file), self.project_dir)
                    module_name = os.path.splitext(module_path.replace(os.path.sep, '.'))[0]
                    self.local_files.add(module_name)

    def _load_module_from_file(self, module_name):
        if module_name in self.module_code:
            return True
        
        module_path = module_name.replace('.', os.path.sep) + '.py'
        full_path = os.path.join(self.project_dir, module_path)
        
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                code = f.read()
            self.add_module(module_name + '.py', code)
            return True
        return False

    def _check_requirements(self):
        to_install: Set[str] = set()

        for module_name in self.dependencies:
            for dep in self.dependencies[module_name]:
                if dep not in self.module_code and dep not in self.local_files:
                    try:
                        importlib.import_module(dep)
                    except ImportError:
                        to_install.add(dep)

        if to_install:
            print(f"The following packages are required but not installed: {', '.join(to_install)}")
            choice = input("Do you want to install these packages automatically? (y/n): ").lower().strip()

            if choice == 'y':
                for package in to_install:
                    print(f"Installing {package}...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                        print(f"Successfully installed {package}")
                    except subprocess.CalledProcessError:
                        print(f"Failed to install {package}. Please install it manually.")
                        sys.exit(1)
            else:
                print("Please install the required packages manually and run the conversion again.")
                sys.exit(1)

        print("All required packages are installed.")
    # def _check_requirements(self):
    #     for module_name in self.dependencies:
            
    #         for dep in self.dependencies[module_name]:
    #             if dep not in self.module_code and dep not in self.local_files:
    #                 try:
    #                     importlib.import_module(dep)
    #                 except ImportError:
    #                     print(f"The package '{dep}' is required but not installed.")
    #                     print("Please install this package in your current environment and run the conversion again.")
    #                     sys.exit(1)
    

    def prepare_execution(self):
        if not hasattr(self, '_prepared') or not self._prepared:
            self._load_local_files()
            self._update_dependencies()
            for module_name in self.dependencies:
                for dep in self.dependencies[module_name]:
                    if dep in self.local_files and dep not in self.module_code:
                        self._load_module_from_file(dep)
                        
            self._update_dependencies()       
            self._check_requirements()
            self._prepared = True

    def execute_module(self, module_name):
        self.prepare_execution()  # Ensure preparation before execution

        if module_name not in self.module_code:
            raise ValueError(f"Module '{module_name}' not found")

        if module_name in self.modules:
            return {"success": True, "output": f"Module '{module_name}' already executed.", "error": None}

        # Create the module object first
        module = types.ModuleType(module_name)
        self.modules[module_name] = module
        sys.modules[module_name] = module

        # Now execute the dependencies
        for dep in self.dependencies[module_name]:
            if dep in self.module_code and dep not in self.modules:
                result = self.execute_module(dep)
                if not result['success']:
                    return result

        code = self.module_code[module_name]

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        redirected_error = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_error

        try:
            # Add project_dir to sys.path to allow imports from the current directory
            if self.project_dir not in sys.path:
                sys.path.insert(0, self.project_dir)

            # Execute code within the module's namespace
            exec(code, module.__dict__)

            # Update globals with the module's namespace
            self.globals.update(module.__dict__)

            return {
                "success": True,
                "output": redirected_output.getvalue(),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": redirected_output.getvalue(),
                "error": f"{str(e)}\n{traceback.format_exc()}"
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def execute_all(self):
        self.prepare_execution()
        results = {}
        for module_name in self.module_code:
            results[module_name] = self.execute_module(module_name)
        return results

    def reset(self):
        self.globals = {'__builtins__': builtins}
        self.modules = {}
        self.module_code = {}
        self.dependencies = defaultdict(set)
        self.local_files = set()
        self._prepared = False  # Reset the preparation flag
        # Remove our custom modules from sys.modules
        for module_name in list(sys.modules.keys()):
            if module_name in self.modules:
                del sys.modules[module_name]