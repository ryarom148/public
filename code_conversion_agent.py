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
        conversation_history = []
        continuation_flag = True

        while continuation_flag:
            system_message = {
                "role": "system",
                "content": "You are responsible for converting SAS code into Python. Use the provided metadata graph to ensure context. Generate the Python code and include comments and explanations within the code. If the output is too long, use continuation prompts to complete the code generation. Ensure only Python code with inline comments is included in the output."
            }
            user_message = {
                "role": "user",
                "content": f"Convert the following SAS code chunk to Python using the provided metadata and context. Provide detailed descriptions and comments within the Python code:\n\nSAS Code:\n{chunk}\n\nMetadata:\n{json.dumps(nx.node_link_data(metadata_graph))}\n\nContext:\n{context}"
            }

            if conversation_history:
                messages = [system_message] + conversation_history
            else:
                messages = [system_message, user_message]

            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=messages,
                max_tokens=1500
            )

            output = response.choices[0].message['content'].strip()
            conversation_history.append({"role": "assistant", "content": output})

            if "Finish" in output:
                continuation_flag = False
                clean_code = output.replace("Finish", "").strip()
                complete_code += clean_code
            else:
                clean_code = output.strip()
                complete_code += clean_code
                conversation_history.append({"role": "user", "content": "continue"})

        return complete_code

    def load_metadata(self, metadata_file):
        return nx.read_gml(metadata_file)

    def chunk_script(self, sas_script):
        lines = sas_script.split('\n')
        chunk_size = 50  # Example chunk size
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
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

# Example usage
api_key = 'your_openai_api_key'
agent = CodeConversionAgent(api_key)

inputs = {
    "sas_folder": "path_to_sas_folder",
    "metadata_file": "metadata.gml"
}
agent.run(inputs)
