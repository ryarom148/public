import openai
from autogen import Agent

class ValidationAgent(Agent):
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

   
class ValidationAgent(Agent):
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def validate_output(self, sas_output, python_output):
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are responsible for validating the converted Python code. Compare the outputs of SAS and Python code, identify discrepancies, and ensure functional consistency."},
                {"role": "user", "content": f"Compare the following outputs and identify discrepancies:\n\nSAS Output:\n{sas_output}\n\nPython Output:\n{python_output}"}
            ],
            max_tokens=1500
        )
        return response.choices[0].message['content']

    def run(self, inputs):
        sas_output_file = inputs['sas_output_file']
        python_output_file = inputs['python_output_file']
        with open(sas_output_file, 'r') as file:
            sas_output = file.read()
        with open(python_output_file, 'r') as file:
            python_output = file.read()
        validation_report = self.validate_output(sas_output, python_output)
        return {"validation_report": validation_report}

# Usage
api_key = 'your_openai_api_key'
sas_output_file = 'path_to_sas_output.txt'
python_output_file = 'path_to_python_output.txt'
agent = ValidationAgent(api_key)
agent.run({"sas_output_file": sas_output_file, "python_output_file": python_output_file})
