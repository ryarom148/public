import json
import networkx as nx
import openai
import os
from autogen import Agent

class SequentialProcessingAgent(Agent):
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.processed_macros = set()  # Track processed macros

    def load_metadata(self, metadata_file):
        return nx.read_gml(metadata_file)

    def update_metadata(self, metadata_graph, script_node, updated_data):
        for key, value in updated_data.items():
            if key in metadata_graph.nodes[script_node]:
                metadata_graph.nodes[script_node][key].update(value)
            else:
                metadata_graph.nodes[script_node][key] = value
        return metadata_graph

    def process_chunk(self, chunk, metadata_graph, script_node, context):
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Your task is to process chunks of SAS code, ensuring logical consistency and data flow. Update the metadata with any new information extracted and provide detailed descriptions of what the code is doing."},
                {"role": "user", "content": f"Process the following SAS code chunk using the provided metadata and context. Provide a detailed description of its functionality and update the metadata:\n\nSAS Code:\n{chunk}\n\nMetadata:\n{json.dumps(nx.node_link_data(metadata_graph))}\n\nContext:\n{context}"}
            ],
            max_tokens=1500
        )

        result = json.loads(response.choices[0].message['content'].strip())
        detailed_description = result['description']
        updated_data = result['metadata']
        
        metadata_graph = self.update_metadata(metadata_graph, script_node, updated_data)
        return metadata_graph, detailed_description, response.choices[0].message['content']

    def process_macro_file(self, macro_node, metadata_graph, context):
        macro_file = metadata_graph.nodes[macro_node]['file_path']
        dependencies = [edge[1] for edge in metadata_graph.edges(macro_node) if metadata_graph.edges[edge]['relationship'] == 'macro_reference']

        # Process dependencies first
        for dependency in dependencies:
            if dependency not in self.processed_macros:
                metadata_graph, context = self.process_macro_file(dependency, metadata_graph, context)

        with open(macro_file, 'r') as file:
            macro_code = file.read()
        chunks = self.chunk_script(macro_code)
        for chunk in chunks:
            metadata_graph, description, context = self.process_chunk(chunk, metadata_graph, macro_node, context)
            # Store detailed description in the metadata
            metadata_graph.nodes[macro_node].setdefault('descriptions', []).append(description)
        
        self.processed_macros.add(macro_node)
        return metadata_graph, context

    def chunk_script(self, sas_script):
        lines = sas_script.split('\n')
        chunk_size = 50  # Example chunk size
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        return ['\n'.join(chunk) for chunk in chunks]

    def process_file(self, file_path, metadata_graph, script_node, context):
        with open(file_path, 'r') as file:
            sas_code = file.read()
        chunks = self.chunk_script(sas_code)
        for chunk in chunks:
            metadata_graph, description, context = self.process_chunk(chunk, metadata_graph, script_node, context)
            # Store detailed description in the metadata
            metadata_graph.nodes[script_node].setdefault('descriptions', []).append(description)
        return metadata_graph, context

    def run(self, inputs):
        sas_folder = inputs['sas_folder']
        metadata_file = inputs['metadata_file']
        data_folder = inputs['data_folder']
        metadata_graph = self.load_metadata(metadata_file)
        
        # Identify all standalone macro files from metadata
        standalone_macros = [node for node, data in metadata_graph.nodes(data=True) if data.get('type') == 'macro']

        context = ""

        # Process each standalone macro file first
        for macro in standalone_macros:
            if macro not in self.processed_macros:
                metadata_graph, context = self.process_macro_file(macro, metadata_graph, context)

        # Process each SAS file
        for sas_file in sorted(os.listdir(sas_folder)):
            if sas_file.endswith('.sas'):
                script_node = sas_file
                file_path = os.path.join(sas_folder, sas_file)
                metadata_graph, context = self.process_file(file_path, metadata_graph, script_node, context)

        self.save_metadata(metadata_graph, metadata_file)
        return {"metadata_file": metadata_file}

    def save_metadata(self, metadata_graph, output_file):
        nx.write_gml(metadata_graph, output_file)
