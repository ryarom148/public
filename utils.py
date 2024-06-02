import os
import re

import os
import re


def load_and_sort_files(folder_path, extension='sas', exclude_subfolders=[]):
    """
    Load files with a specific extension from the specified folder and sort them in ascending order.
    Sorting is done based on numerical order if numbers are present at the beginning or end of the filename,
    otherwise alphabetical order. Only files with the specified extension are considered.

    Args:
        folder_path (str): The path to the folder containing the files.
        extension (str): The file extension to filter by (default is 'sas').
        exclude_subfolders (list): List of subfolder names to exclude from processing (default is []).

    Returns:
        list: A sorted list of file paths.
    """
    def extract_sort_key(filename):
        base_name = filename.lower().rsplit('.', 1)[0]

        # Extract parts from the filename considering "_" and "."
        parts = re.split(r'[_\.]', base_name)

        # Function to convert parts into sorting keys
        def numeric_or_string(value):
            return int(value) if value.isdigit() else value

        # Separate numbers at the beginning and end, and other parts
        sort_key = []
        if parts[0].isdigit():
            sort_key.append(numeric_or_string(parts[0]))
            parts = parts[1:]
        if parts and parts[-1].isdigit():
            last_part = parts.pop()
        else:
            last_part = None

        sort_key.extend(numeric_or_string(part) for part in parts)

        if last_part is not None:
            sort_key.append(numeric_or_string(last_part))

        return tuple(sort_key)

    # Collect all files with the specified extension, excluding specified subfolders
    matched_files = []
    for root, dirs, files in os.walk(folder_path):
        # Remove excluded subfolders from dirs to prevent os.walk from traversing them
        dirs[:] = [d for d in dirs if os.path.relpath(os.path.join(root, d), folder_path) not in exclude_subfolders]
        matched_files.extend(
            os.path.join(root, file) for file in files if file.lower().endswith(f'.{extension}')
        )

    # Sort files based on their sort key
    sorted_files = sorted(matched_files, key=lambda file: extract_sort_key(os.path.basename(file)))

    return sorted_files

# Example usage:
#folder_path = 'path/to/your/folder'
#exclude_subfolders = ['path/to/your/folder/macros']
#sorted_files = load_and_sort_files(folder_path, extension='sas', exclude_subfolders=exclude_subfolders)
#print(sorted_files)

import re
import json

def extract_code_blocks_and_info(text):
    # Regex pattern to match the code blocks
    code_block_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    code_blocks = code_block_pattern.findall(text)

    # Regex pattern to match the metadata for modules
    metadata_pattern = re.compile(
        r'### name (?:"(.*?)"|(\w+))\n'
        r'### file_path: "(.*?)"\n'
        r'### calling external macros : \[(.*?)\]\n'
        r'### who is calling : \[(.*?)\]\n'
        r'### input data sourcse: \[(.*?)\]\n'
        r'### output data sources: \[(.*?)\]\n'
    )

    # Regex pattern to match the metadata for macros
    macro_pattern = re.compile(
        r'### name "(.*?)"\n'
        r'### file_path: "(.*?)"\n'
        r'### calling external macros: \[(.*?)\]\n'
        r'### who is calling : \[(.*?)\]\n'
        r'### functions and methods signature \[(.*?)\]'
    )

    # Extract metadata
    metadata_matches = metadata_pattern.findall(text)
    macro_matches = macro_pattern.findall(text)

    modules = []
    macros = []

    for match in metadata_matches:
        name = match[0] if match[0] else match[1]
        file_path = match[2]
        calling_macros = match[3]
        who_is_calling = match[4]
        input_sources = match[5]
        output_sources = match[6]

        modules.append({
            "name": name.strip(),
            "file_path": file_path.strip(),
            "calling_external_macros": [m.strip() for m in calling_macros.split(',')],
            "who_is_calling": [m.strip() for m in who_is_calling.split(',')],
            "input_data_sources": [m.strip().strip("'") for m in input_sources.split(',')],
            "output_data_sources": [m.strip().strip("'") for m in output_sources.split(',')]
        })

    for match in macro_matches:
        name = match[0]
        file_path = match[1]
        calling_macros = match[2]
        who_is_calling = match[3]
        function_signatures = match[4]
        
        function_signatures_list = []
        function_signatures = function_signatures.split('},')
        for function_signature in function_signatures:
            signature_match = re.search(r'{(.*?), callers \[(.*?)\]}', function_signature.strip())
            if signature_match:
                signature, callers = signature_match.groups()
                function_signatures_list.append({
                    "signature": signature.strip(),
                    "callers": [c.strip() for c in callers.split(',')]
                })
        
        macros.append({
            "name": name.strip(),
            "file_path": file_path.strip(),
            "calling_external_macros": [m.strip() for m in calling_macros.split(',')],
            "who_is_calling": [m.strip() for m in who_is_calling.split(',')],
            "functions_and_methods_signatures": function_signatures_list
        })

    return {
        "code_blocks": code_blocks,
        "modules": modules,
        "macros": macros
    }

# Sample text
text = """
```python
### name Module1
### file_path: "macros/Macro.sas"
### calling external macros : [macro1, macro2]
### who is calling : [module4, module6]
### input data sourcse: ['data/file1.xlsx','data/file2.xls']
### output data sources: ['data/output/output1.csv','data/output/output1.xls']
import pandas as pd
from macros.macro1 import method1
from macros.macro2 import method1
'''
# SAS Code Extract:
data example;
    set data_source;
    if condition then output;
run;
'''

'''
This SAS code creates a dataset called 'example' by reading from 'data_source'. It checks a specified condition for each row, and if the condition is true, the row is included in the output dataset.
'''

# Load the data
data_source = pd.read_csv('../data/data_source.csv')

# Apply the condition and output the result
example = data_source[data_source['condition']]
example.to_csv('../output_code/example.csv', index=False)

# Note: This output is saved because it is an in-memory dataset, modified from the original SAS logic.

# name of sas module is done
```
```python
### name "Macro1"
### file_path: "macros/Macro.sas"
### calling external macros: [macro5, macro6]
### who is calling : [module1, module2]
### functions and methods signature [{signature1, callers [moduel1, module2]} ,[ signature2, callers[module7,module1]]
import pandas as pd
from macros.macro1 import method1

'''
# SAS Code Extract:
data example;
    set data_source;
    if condition then output;
run;
'''

'''
This SAS code creates a dataset called 'example' by reading from 'data_source'. It checks a specified condition for each row, and if the condition is true, the row is included in the output dataset.
'''

# Load the data
data_source = pd.read_csv('../data/data_source.csv')

# Apply the condition and output the result
example = data_source[data_source['condition']]
example.to_csv('../output_code/example.csv', index=False)

# Note: This output is saved because it is an in-memory dataset, modified from the original SAS logic.

# name of sas module is done
```
"""

result = extract_code_blocks_and_info(text)
print(json.dumps(result, indent=4))
