import re
import os
from rich import print
from collections import defaultdict
from typing import Dict, List, Optional, Any
class SASRegexPatterns:
    SINGLE_LINE_COMMENT = re.compile(r'^\s*\*.*?;$', re.MULTILINE)
    BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
    CALL_SYMPUT = re.compile(r'call\s+symput\s*\(\s*[\'"](\w+)[\'"]\s*,\s*(&?\w+|\'[^\']+\')\s*\);', re.IGNORECASE)
    CALL_SYMPUTX = re.compile(r'call\s+symputx\s*\(\s*[\'"](\w+)[\'"]\s*,\s*(&?\w+|\'[^\']+\')\s*,?\s*[\'"]?[L]?[\'"]?\s*\);', re.IGNORECASE)
    CALL_SYMGET = re.compile(r'\b(\w+)\s*=\s*symget\s*\(\s*[\'"](\w+)[\'"]\s*\);', re.IGNORECASE)
    CALL_SYMGETN = re.compile(r'\b(\w+)\s*=\s*symgetn\s*\(\s*[\'"](\w+)[\'"]\s*\);', re.IGNORECASE)
    GLOBAL = re.compile(r'%global\s+([\w\s]+);', re.IGNORECASE)
    LET = re.compile(r'%let\s+(&?\w+)\s*=\s*([^;]+);', re.IGNORECASE)
    STRING = re.compile(r'(["\'])(.*?)(\1)', re.IGNORECASE)
    STRING_VAR = re.compile(r'&(\w+)(?=\.)|&(\w+)(?=$)')
    DOUBLE_STRING_VAR = re.compile(r'&&(\w+)(?=\.)|&(\w+)(?=$)')
    PROC = re.compile(r'^\s*proc\s+(\w+)', re.MULTILINE | re.IGNORECASE)
    DATA_STEP = re.compile(r'^\s*data\s+(\w+)', re.MULTILINE | re.IGNORECASE)
    DATA_STEP_SQL = re.compile(r'^\s*create\s+table\s+(\w+)', re.MULTILINE | re.IGNORECASE)
    LIBRARY_REF = re.compile(r'\b(\w+)\.\w+', re.IGNORECASE)
    INCLUDE = re.compile(r'%include\s+([\'"])(.+?\.sas)\1;', re.IGNORECASE)

       

#Create a new class called `SASRegexOperations` to handle regex-related operations:
class SASRegexOperations:
    @staticmethod
    def remove_comments(sas_code):
        sas_code = re.sub(SASRegexPatterns.BLOCK_COMMENT, '', sas_code)
        sas_code = re.sub(SASRegexPatterns.SINGLE_LINE_COMMENT, '', sas_code)
        return sas_code

    @staticmethod
    def find_global_variables(sas_code):
        global_vars = SASRegexPatterns.GLOBAL.findall(sas_code)
        return {var.strip('&').strip() for var_list in global_vars for var in var_list.split()}

    @staticmethod
    def find_procedure_statements(code):
        return SASRegexPatterns.PROC.findall(code)

    @staticmethod
    def find_data_step_definitions(code):
        return SASRegexPatterns.DATA_STEP.findall(code)

    @staticmethod
    def find_library_references(code):
        library_refs = SASRegexPatterns.LIBRARY_REF.findall(code)
        return list(set(library_refs))

    @staticmethod
    def find_external_macros(sas_code):
        file_refs = SASRegexPatterns.INCLUDE.findall(sas_code)
        processed_refs = []
        for _, path in file_refs:
            if 'macro' in path.lower():
                processed_refs.append(path)
            else:
                filename = os.path.basename(path)
                processed_refs.append(filename)
        return processed_refs
    
class SASParserError(Exception):
    def __init__(self, message, module=None, line_number=None):
        self.message = message
        self.module = module
        self.line_number = line_number
        super().__init__(self.format_error())

    def format_error(self):
        error_msg = f"SASParser Error: {self.message}"
        if self.module:
            error_msg += f" in module '{self.module}'"
        if self.line_number:
            error_msg += f" at line {self.line_number}"
        return error_msg

class SASParser:
    """
    A tool for parsing and analyzing SAS code modules, designed for use with LLM agents.

    This parser extracts metadata from SAS code, including global and local variables,
    variable usage, external file references, and more. It's particularly useful for
    understanding the structure and dependencies of SAS programs before conversion
    to other languages or for code analysis tasks.

    When to use:
    - Before converting SAS code to another language (e.g., Python)
    - When analyzing dependencies between SAS modules
    - To understand variable usage and scope across multiple SAS files
    - For extracting metadata about SAS procedures and data steps

    Example usage:
    ```python
    # Define your SAS modules
    sas_modules = {
        'module1.sas': '''
            %global global_var1;
            %let local_var = 5;
            data output;
                set input;
                global_var1 = local_var * 2;
            run;
        ''',
        'module2.sas': '''
            %include 'module1.sas';
            proc print data=output;
            run;
        '''
    }

    # Create a SASParser instance
    parser = SASParser(sas_modules)

    # Get metadata for a specific module
    metadata = parser.get_module_comprehensive_metadata('module1.sas')
    print(metadata)

    # Get metadata for all modules
    all_metadata = parser.get_all_metadata()
    print(all_metadata)

    # The metadata includes information about:
    # - Global and local variables
    # - Variable usage
    # - Import statements (showing where variables are defined)
    # - External file references
    # - PROC statements and data steps
    # - Library references
    # - Any warnings or errors encountered during parsing
    ```

    Note: This parser uses regex patterns to extract information from SAS code.
    While it covers many common SAS constructs, it may not capture every possible
    SAS syntax variation. Always verify the results for complex SAS programs.
    """
    def __init__(self, sas_modules: Dict[str, str]):
        self.sas_modules = sas_modules
        self.regex_ops = SASRegexOperations()
        self.warnings: List[Dict[str, Optional[str]]] = []
        self.all_global_vars = self._extract_all_global_variables()
        self.variable_map = self._create_variable_usage_map()
        self.last_assignments = {module: self._track_variable_assignments_by_module(module) for module in sas_modules}
        self.external_files = self._extract_external_file_references() 
        self.meta_data = self.get_all_metadata()

    def get_module_comprehensive_metadata(self, module_name: str, variables_info: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        self.warnings = []  # Reset warnings for this module
        if module_name not in self.sas_modules:
            raise SASParserError(f"Module '{module_name}' not found")

        try:
            code = self.sas_modules[module_name]
            import_info = self._get_variable_import_information(module_name, variables_info)

            metadata = {
                "module_name": module_name,
                "metadata": {
                    "global": import_info['global'],
                    "local": import_info['local'],
                    "used": import_info['used'],
                    "imports": import_info['imports'],
                    "external_files": import_info['external_files'],
                    "proc_statements": self.regex_ops.find_procedure_statements(code),
                    "data_steps": self.regex_ops.find_data_step_definitions(code),
                    "library_references": self.regex_ops.find_library_references(code)
                },
                "warnings": self.warnings
            }
            return metadata
        except Exception as e:
            raise SASParserError(f"Error processing module: {str(e)}", module=module_name)

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        all_metadata = {}
        for module in self.sas_modules:
            try:
                all_metadata[module] = self.get_module_comprehensive_metadata(module)
            except SASParserError as e:
                all_metadata[module] = {"error": str(e)}
        return all_metadata


    def _get_line_number(self, code, position):
        return code[:position].count('\n') + 1

    def _add_warning(self, message, module=None, line_number=None):
        warning = {
            "message": message,
            "module": module,
            "line_number": line_number
        }
        self.warnings.append(warning)
    def _extract_external_file_references(self):
        all_external_files = {}
        for module_name, sas_code in self.sas_modules.items():
            try:
                _code = self.regex_ops.remove_comments(sas_code)
                files_found = self.regex_ops.find_external_macros(_code)
                if files_found:
                    all_external_files[module_name] = files_found
            except Exception as e:
                self._add_warning(f"Error extracting external file references: {str(e)}", module_name)
        return all_external_files
    def _extract_all_global_variables(self):
        all_global_vars = set()
        for module_name, sas_code in self.sas_modules.items():
            try:
                globals_found = self._extract_global_variables(sas_code, module_name)
                all_global_vars.update(globals_found)
            except SASParserError as e:
                self._add_warning(str(e), module_name)
        return all_global_vars

    def _extract_global_variables(self, sas_code, module_name):
        try:
            sas_code = self.regex_ops.remove_comments(sas_code)
            return self.regex_ops.find_global_variables(sas_code)
        except Exception as e:
            raise SASParserError(f"Error extracting global variables: {str(e)}", module=module_name)
    
    def _strip_ampersands(self,var: str) -> str:
        return var.lstrip('&')
    
    def _extract_local_and_reclassified_globals(self, sas_code, module_name):
        try:
            sas_code = self.regex_ops.remove_comments(sas_code)
            call_symput_vars = SASRegexPatterns.CALL_SYMPUT.findall(sas_code)
            call_symputx_vars = SASRegexPatterns.CALL_SYMPUTX.findall(sas_code)
            let_vars = SASRegexPatterns.LET.findall(sas_code)
            
            local_vars = set()
            _globals = self._extract_global_variables(sas_code, module_name)
            
            for var, _ in call_symput_vars + call_symputx_vars + let_vars:
                var = self._strip_ampersands(var).strip()
                if not var.isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")):
                    if var in self.all_global_vars:
                        _globals.add(var)
                    else:
                        local_vars.add(var)

            return _globals, local_vars
        except Exception as e:
            raise SASParserError(f"Error extracting local and reclassified global variables: {str(e)}", module=module_name)

    def _extract_used_variables(self, sas_code, local_vars, global_vars, module_name):
        try:
            sas_code = self.regex_ops.remove_comments(sas_code)
            used_vars = set()

            call_symput_matches = SASRegexPatterns.CALL_SYMPUT.findall(sas_code)
            call_symputx_matches = SASRegexPatterns.CALL_SYMPUTX.findall(sas_code)
            used_vars.update(self._strip_ampersands(var).strip() for _, var in call_symput_matches + call_symputx_matches if not self._strip_ampersands(var).strip().isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")) and self._strip_ampersands(var).strip() not in local_vars)

            call_symget_matches = SASRegexPatterns.CALL_SYMGET.findall(sas_code)
            call_symgetn_matches = SASRegexPatterns.CALL_SYMGETN.findall(sas_code)
            used_vars.update(self._strip_ampersands(var).strip() for _, var in call_symget_matches + call_symgetn_matches if not self._strip_ampersands(var).strip().isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")) and self._strip_ampersands(var).strip() not in local_vars)

            string_matches = SASRegexPatterns.STRING.findall(sas_code)
            for _, string, _ in string_matches:
                vars_in_string = SASRegexPatterns.STRING_VAR.findall(string)
                double_vars_in_string = SASRegexPatterns.DOUBLE_STRING_VAR.findall(string)
                used_vars.update(self._strip_ampersands(var).strip() for match in vars_in_string for var in match if var and not self._strip_ampersands(var).strip().isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")) and self._strip_ampersands(var).strip() not in local_vars)
                used_vars.update(self._strip_ampersands(var).strip() for match in double_vars_in_string for var in match if var and not self._strip_ampersands(var).strip().isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")) and self._strip_ampersands(var).strip() not in local_vars)
        
            let_matches = SASRegexPatterns.LET.findall(sas_code)
            used_vars.update(self._strip_ampersands(var).strip() for var, _ in let_matches if not self._strip_ampersands(var).strip().isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")) and self._strip_ampersands(var).strip() not in local_vars)
            used_vars.update(self._strip_ampersands(assigned_value).strip() for _, assigned_value in let_matches if not self._strip_ampersands(assigned_value).strip().isdigit() and not (assigned_value.startswith('"') and assigned_value.endswith('"')) and not (assigned_value.startswith("'") and assigned_value.endswith("'")) and self._strip_ampersands(assigned_value).strip() not in local_vars)

            used_globals = {var for var in used_vars if var in global_vars}
            global_vars.update(used_globals)
            used_vars -= used_globals

            return used_vars
        except Exception as e:
            raise SASParserError(f"Error extracting used variables: {str(e)}", module=module_name)

    def _extract_external_file_references(self):
        all_external_files = {}
        for module_name, sas_code in self.sas_modules.items():
            try:
                _code = self.regex_ops.remove_comments(sas_code)
                files_found = self.regex_ops.find_external_macros(_code)
                if files_found:
                    all_external_files[module_name] = files_found
            except Exception as e:
                self._add_warning(f"Error extracting external file references: {str(e)}", module_name)
        return all_external_files

    def _create_variable_usage_map(self):
        variable_map = {}
        for module_name, sas_code in self.sas_modules.items():
            try:
                _globals, locals_found = self._extract_local_and_reclassified_globals(sas_code, module_name)
                used_vars = self._extract_used_variables(sas_code, locals_found, _globals, module_name)
                variable_map[module_name] = {
                    'global': list(_globals),
                    'local': list(locals_found),
                    'used': list(used_vars)
                }
            except SASParserError as e:
                self._add_warning(str(e), module_name)
                variable_map[module_name] = {
                    'global': [],
                    'local': [],
                    'used': []
                }
        return variable_map

    def _track_variable_assignments_by_module(self, module_name):
        try:
            last_assignment = {}
            module_names = list(self.sas_modules.keys())
            index = module_names.index(module_name)
            
            for i in range(index):
                current_module_name = module_names[i]
                sas_code = self.regex_ops.remove_comments(self.sas_modules[current_module_name])
                let_matches = SASRegexPatterns.LET.findall(sas_code)
                call_symput_matches = SASRegexPatterns.CALL_SYMPUT.findall(sas_code)
                call_symputx_matches = SASRegexPatterns.CALL_SYMPUTX.findall(sas_code)
                global_matches = SASRegexPatterns.GLOBAL.findall(sas_code)

                for var, _ in let_matches + call_symput_matches + call_symputx_matches:
                    var = self._strip_ampersands(var).strip()
                    if not var.isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")):
                        last_assignment[var] = current_module_name

                for var_list in global_matches:
                    for var in var_list.split():
                        var = self._strip_ampersands(var).strip()
                        if not var.isdigit() and not (var.startswith('"') and var.endswith('"')) and not (var.startswith("'") and var.endswith("'")):
                            last_assignment[var] = current_module_name

            return last_assignment
        except Exception as e:
            raise SASParserError(f"Error tracking variable assignments: {str(e)}", module=module_name)

    def _get_variable_import_information(self, module_name, variables_info=None):
        try:
            if variables_info is None:
                variables_list = list(set(self.variable_map[module_name]['global'] + 
                                          self.variable_map[module_name]['local'] + 
                                          self.variable_map[module_name]['used']))
            else:
                variables_list = variables_info['variables']

            globals_in_module = set()
            locals_in_module = set()
            used_in_module = set()

            for var in variables_list:
                if var in self.variable_map[module_name]['global']:
                    globals_in_module.add(var)
                elif var in self.variable_map[module_name]['local']:
                    locals_in_module.add(var)
                elif var in self.all_global_vars:
                    globals_in_module.add(var)
                else:
                    used_in_module.add(var)

            import_statements = {'global': {}, 'local_Used': {}}
            last_assignment = self.last_assignments[module_name]

            for var in globals_in_module:
                if var in last_assignment and last_assignment[var] != module_name:
                    if last_assignment[var] not in import_statements['global']:
                        import_statements['global'][last_assignment[var]] = []
                    import_statements['global'][last_assignment[var]].append(var)

            for var in locals_in_module | used_in_module:
                if var in last_assignment and last_assignment[var] != module_name:
                    if last_assignment[var] not in import_statements['local_Used']:
                        import_statements['local_Used'][last_assignment[var]] = []
                    import_statements['local_Used'][last_assignment[var]].append(var)

            external_files_imports = self._get_external_files_imports(module_name, variables_info)

            return {
                'global': list(globals_in_module),
                'local': list(locals_in_module),
                'used': list(used_in_module),
                'imports': import_statements,
                'external_files': external_files_imports
            }
        except Exception as e:
            raise SASParserError(f"Error getting variable import information: {str(e)}", module=module_name)

    def _get_external_files_imports(self, module_name, variables_info=None):
        external_files_imports = {}
        external_files = self.external_files.get(module_name, [])
        
        for external_file in external_files:
            if external_file in self.sas_modules:
                if variables_info and 'external' in variables_info and external_file in variables_info['external']:
                    variables_in_file = variables_info['external'][external_file]
                else:
                    external_code = self.sas_modules[external_file]
                    globals_in_file, locals_in_file = self._extract_local_and_reclassified_globals(external_code, external_file)
                    used_in_file = self._extract_used_variables(external_code, locals_in_file, globals_in_file, external_file)
                    variables_in_file = list(globals_in_file | locals_in_file | used_in_file)

                imports_for_file = {'global': {}, 'local_Used': {}}
                
                for var in variables_in_file:
                    if var in self.last_assignments[module_name]:
                        last_assigned_module = self.last_assignments[module_name][var]
                        if last_assigned_module != external_file:
                            if var in globals_in_file:
                                if last_assigned_module not in imports_for_file['global']:
                                    imports_for_file['global'][last_assigned_module] = []
                                imports_for_file['global'][last_assigned_module].append(var)
                            else:
                                if last_assigned_module not in imports_for_file['local_Used']:
                                    imports_for_file['local_Used'][last_assigned_module] = []
                                imports_for_file['local_Used'][last_assigned_module].append(var)
                
                external_files_imports[external_file] = {
                    'variables': variables_in_file,
                    'imports': imports_for_file
                }
        
        return external_files_imports
