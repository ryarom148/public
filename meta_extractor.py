import os
import json
import networkx as nx
import openai
from autogen import Agent

class MetadataExtractionAgent(Agent):
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def extract_metadata(self, sas_script, file_path):
        metadata_graph = nx.DiGraph()
        sas_code = ""

        with open(sas_script, 'r') as file:
            sas_code = file.read()

        system_prompt = """
        You are responsible for parsing SAS scripts to extract metadata. Extract global variables, local variables, macro definitions, their parameters, data sources, file locations, code block positions, parameter flows to macros, serialized file outputs, in-memory data outputs, and relationships between scripts. Use the following template for the metadata:
        {
            "nodes": [
                {
                    "id": "script1.sas",
                    "type": "script",
                    "file_path": "/path/to/script1.sas",
                    "global_variables": ["var1", "var2"],
                    "local_variables": ["local1", "local2"],
                    "macros": [
                        {
                            "name": "macro1",
                            "parameters": ["param1", "param2"],
                            "start_row": 10,
                            "end_row": 20,
                            "file_path": "/path/to/macros/macro1.sas"
                        }
                    ],
                    "data_sources": [
                        {
                            "name": "data1",
                            "file_path": "/data_folder/data1.csv"
                        }
                    ],
                    "output_files": [
                        {
                            "name": "output1",
                            "file_path": "/output_folder/output1.csv"
                        }
                    ],
                    "in_memory_outputs": ["dataset1"]
                }
            ],
            "edges": [
                {
                    "source": "script1.sas",
                    "target": "script2.sas",
                    "relationship": "data_flow",
                    "data_passed": ["output1.csv"]
                },
                {
                    "source": "script1.sas",
                    "target": "macro1",
                    "relationship": "macro_reference",
                    "file_path": "/path/to/macros/macro1.sas"
                }
            ]
        }
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract metadata from the following SAS code located at {file_path}:\n\n{sas_code}"}
            ],
            max_tokens=1500
        )

        metadata = json.loads(response.choices[0].message['content'].strip())
        
        for key, value in metadata.items():
            value['file_path'] = file_path
            metadata_graph.add_node(key, value=value)

        return metadata_graph

    def process_folder(self, folder_path, parent_graph=None):
        metadata_graph = parent_graph if parent_graph else nx.DiGraph()

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.sas'):
                    file_path = os.path.join(root, file)
                    file_metadata = self.extract_metadata(file_path, file_path)
                    metadata_graph = nx.compose(metadata_graph, file_metadata)

        return metadata_graph

    def save_metadata(self, metadata_graph, output_file):
        nx.write_gml(metadata_graph, output_file)

    def run(self, inputs):
        sas_folder = inputs['sas_folder']
        metadata_output = inputs['metadata_output']
        all_metadata = self.process_folder(sas_folder)
        self.save_metadata(all_metadata, metadata_output)
        return {"metadata_file": metadata_output}
