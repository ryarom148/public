import networkx as nx
from autogen import Agent

class ReportingAgent(Agent):
    def __init__(self, api_key):
        self.api_key = api_key

    def load_metadata(self, metadata_file):
        return nx.read_gml(metadata_file)

    def generate_report(self, metadata_graph):
        report = "# SAS to Python Conversion Report\n\n"
        report += "## Summary\n\n"
        report += "This report provides an overview of the SAS scripts converted to Python, detailing the functionality, inputs, and outputs of each script.\n\n"

        for node in metadata_graph.nodes(data=True):
            report += f"### {node[0]}\n\n"
            report += f"**Description**: {node[1].get('description', 'No description available')}\n\n"
            report += f"**Inputs**: {node[1].get('inputs', 'No inputs available')}\n\n"
            report += f"**Outputs**: {node[1].get('outputs', 'No outputs available')}\n\n"
            report += f"**Logic**: {node[1].get('logic', 'No logic available')}\n\n"
            report += f"**Code Block Position**: Start Row: {node[1].get('start_row', 'N/A')}, End Row: {node[1].get('end_row', 'N/A')}\n\n"
            report += "---\n\n"

        # Generate network diagram
        report += "## Relationships and Parameters\n\n"
        report += "The following diagram depicts the relationships and parameter flows between the scripts:\n\n"
        # Here you would include logic to generate and insert a network diagram image into the report.

        return report

    def save_report(self, report, output_file):
        with open(output_file, 'w') as file:
            file.write(report)

    def run(self, inputs):
        metadata_file = inputs['metadata_file']
        report_output = inputs['report_output']
        metadata_graph = self.load_metadata(metadata_file)
        report = self.generate_report(metadata_graph)
        self.save_report(report, report_output)
        return {"report_file": report_output}

# Usage
metadata_file = 'metadata.gml'
report_output = 'conversion_report.md'
api_key = 'your_openai_api_key'
agent = ReportingAgent(api_key)
agent.run({"metadata_file": metadata_file, "report_output": report_output})
