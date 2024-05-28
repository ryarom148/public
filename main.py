from autogen import AutoGenFramework

class AutomationFrameworkAgent:
    def __init__(self, metadata_agent, processing_agent, conversion_agent, validation_agent, reporting_agent):
        self.metadata_agent = metadata_agent
        self.processing_agent = processing_agent
        self.conversion_agent = conversion_agent
        self.validation_agent = validation_agent
        self.reporting_agent = reporting_agent

    def run(self, sas_folder, metadata_file, api_key, sas_output_file, python_output_file, report_output, data_folder):
        print("Extracting metadata...")
        metadata_result = self.metadata_agent.run({"sas_folder": sas_folder, "metadata_output": metadata_file})
        
        print("Processing SAS scripts sequentially...")
        processing_result = self.processing_agent.run({
            "sas_folder": sas_folder,
            "metadata_file": metadata_result["metadata_file"],
            "data_folder": data_folder
        })
        
        print("Converting SAS scripts to Python...")
        conversion_result = self.conversion_agent.run({
            "sas_folder": sas_folder,
            "metadata_file": processing_result["metadata_file"]
        })
        
        print("Validating converted Python code...")
        validation_result = self.validation_agent.run({
            "sas_output_file": sas_output_file,
            "python_output_file": python_output_file
        })

        print("Generating report...")
        report_result = self.reporting_agent.run({
            "metadata_file": processing_result["metadata_file"],
            "report_output": report_output
        })

        print("Conversion process completed successfully.")
        return report_result

# Initialize agents
metadata_agent = MetadataExtractionAgent(api_key='your_openai_api_key')
processing_agent = SequentialProcessingAgent(api_key='your_openai_api_key')
conversion_agent = CodeConversionAgent(api_key='your_openai_api_key')
validation_agent = ValidationAgent(api_key='your_openai_api_key')
reporting_agent = ReportingAgent(api_key='your_openai_api_key')

# Initialize and configure AutoGen framework
framework = AutoGenFramework()

framework.add_agent("metadata_extraction", metadata_agent)
framework.add_agent("sequential_processing", processing_agent)
framework.add_agent("code_conversion", conversion_agent)
framework.add_agent("validation", validation_agent)
framework.add_agent("reporting", reporting_agent)

# Define workflow
framework.define_workflow([
    {"agent": "metadata_extraction", "inputs": {"sas_folder": 'path_to_sas_folder', "metadata_output": 'metadata.gml'}, "outputs": ["metadata_file"]},
    {"agent": "sequential_processing", "inputs": {"sas_folder": 'path_to_sas_folder', "metadata_file": "metadata_file", "data_folder": "path_to_data_folder"}, "outputs": ["metadata_file"]},
    {"agent": "code_conversion", "inputs": {"sas_folder": 'path_to_sas_folder', "metadata_file": "metadata_file"}, "outputs": ["python_files"]},
    {"agent": "validation", "inputs": {"sas_output_file": 'expected_sas_output.txt', "python_output_file": 'generated_python_output.txt'}, "outputs": ["validation_report"]},
    {"agent": "reporting", "inputs": {"metadata_file": "metadata_file", "report_output": 'conversion_report.md'}, "outputs": ["report_file"]}
])

# Run workflow
results = framework.run({
    "sas_folder": 'path_to_sas_folder',
    "metadata_output": 'metadata.gml',
    "sas_output_file": 'expected_sas_output.txt',
    "python_output_file": 'generated_python_output.txt',
    "report_output": 'conversion_report.md',
    "data_folder": 'path_to_data_folder'
})

print(results)
