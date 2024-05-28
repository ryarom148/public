import openai
import networkx as nx
import json
import os
from autogen import Agent

class CodeConversionAgent(Agent):
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def convert_chunk_to_python(self, chunk, metadata_graph, context):
        complete_code = ""
        continuation_flag = True

        while continuation_flag:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are responsible for converting SAS code into Python. Use the provided metadata graph to ensure context. Extract SAS code segments, describe the logic, and generate equivalent Python code. If the output is too long, use 'continue' prompts to complete the code. Ensure no comments are included, only the code."},
                    {"role": "user", "content": f"Convert the following SAS code chunk to Python using the provided metadata and context. Provide detailed descriptions of the logic:\n\nSAS Code:\n{chunk}\n\nMetadata:\n{json.dumps(nx.node_link_data(metadata_graph))}\n\nContext:\n{context}"}
                ],
                max_tokens=1500
            )

            output = response.choices[0].message['content'].strip()

            if "finished" in output.lower():
                continuation_flag = False
                clean_code = output.replace("finished", "").strip()
                complete_code += clean_code
            else:
                clean_code = output.strip()
                complete_code += clean_code

        return complete_code

    def load_metadata(self, metadata_file):
        return nx.read_gml(metadata_file)

    def chunk_script(self, sas_script):
        lines = sas_script.split('\n')
        chunk_size = 50  # Example chunk size
        chunks = [lines[i[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        return ['\n'.join(chunk) for chunk in chunks]

    def run(self, inputs):
        sas_folder = inputs['sas_folder']
        metadata_file = inputs['metadata_file']
        metadata_graph = self.load_metadata(metadata_file)
        
        complete_python_code = ""
        context = ""

        for sas_file in sorted(os.listdir(sas_folder)):
            if sas_file.endswith('.sas'):
                with open(os.path.join(sas_folder, sas_file), 'r') as file:
                    sas_code = file.read()
                chunks = self.chunk_script(sas_code)
                for chunk in chunks:
                    python_code = self.convert_chunk_to_python(chunk, metadata_graph, context)
                    complete_python_code += python_code + "\n"
                output_file = os.path.join(sas_folder, sas_file.replace('.sas', '.py'))
                with open(output_file, 'w') as py_file:
                    py_file.write(complete_python_code)
        return {"python_files": [file for file in os.listdir(sas_folder) if file.endswith('.py')]}
