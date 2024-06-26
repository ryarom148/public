import asyncio
from typing import Dict, List, Any, Optional
import openai
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.status import Status
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.box import ROUNDED
from rich.markdown import Markdown
from rich.panel import Panel
import time
from itertools import zip_longest
from sas_parcer import SASParser
console = Console()

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time:
            self.elapsed = time.time() - self.start_time
            self.start_time = None

class BaseAgent:
    def __init__(self, name: str, create_subagents: bool = False, model: str = "gpt-4", debug: bool = False):
        self.name = name
        self.create_subagents = create_subagents
        self.model = model
        self.debug = debug
        self.tools = {}
        self.conversation_history = []
        self.subagents = []

    def register_tool(self, tool: callable):
        self.tools[tool.__name__] = {"function": tool, "description": tool.__doc__}

    async def use_tool(self, tool_name: str, *args, **kwargs):
        if tool_name in self.tools:
            return await self.tools[tool_name]["function"](*args, **kwargs)
        else:
            raise ValueError(f"Tool {tool_name} not found")

    def add_to_history(self, message: Dict[str, str]):
        self.conversation_history.append(message)

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def get_last_message(self) -> Optional[Dict[str, str]]:
        return self.conversation_history[-1] if self.conversation_history else None

    async def generate_response(self, prompt: str, guided_json: bool = False, task_description: str = "Generating") -> str:
        self.add_to_history({"role": "user", "content": prompt})
        
        full_prompt = prompt
        if self.tools:
            tools_prompt = "\n".join([f"{name}: {spec['description']}" for name, spec in self.tools.items()])
            full_prompt += f"\n\nAvailable tools:\n{tools_prompt}\n\nDecide if you need to use any tools to complete the task."
        
        start_time = time.time()
        
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
            task = progress.add_task(f"[cyan]{self.name}: {task_description}", total=None)
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=self.conversation_history + [{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"} if guided_json else None
            )
        
        elapsed_time = time.time() - start_time
        content = response.choices[0].message['content']
        self.add_to_history({"role": "assistant", "content": content})
        
        usage = response.usage
        console.print(f"{self.name} ({task_description}): {usage.total_tokens} tokens, {elapsed_time:.2f}s")
        
        if self.debug:
            console.print(Panel(content, title=f"[bold green]{self.name} Response ({task_description})[/bold green]", expand=False))
        
        if "FINISH_DONE" not in content:
            return await self.generate_response("Continue generation", guided_json, task_description)
        
        return content.replace("FINISH_DONE", "").strip()

    async def create_subagent(self, agent_class, *args, **kwargs):
        if self.create_subagents:
            subagent = agent_class(*args, **kwargs)
            self.subagents.append(subagent)
            return subagent
        else:
            raise ValueError("This agent is not allowed to create subagents")

class MetadataExtractionAgent(BaseAgent):
   def __init__(self, sas_modules: Dict[str, str], model: str = "gpt-4", debug: bool = False):
        super().__init__("MetadataExtractor", create_subagents=False, model=model, debug=debug)
        self.sas_modules = sas_modules
        self.parser = SASParser(sas_modules)
        self.metadata = {}
        asyncio.run(self.extract_metadata())

        async def extract_metadata(self):
            tasks = []
            for module_name, metadata in self.parser.meta_data.items():
                if "macro" not in module_name.lower():
                    task = self.process_module(module_name, metadata)
                    tasks.append(task)
            
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=not self.debug) as progress:
                task = progress.add_task("[cyan]Extracting metadata", total=len(tasks))
                results = await asyncio.gather(*tasks)
                progress.update(task, advance=1)
            
            for module_name, result in zip(self.parser.meta_data.keys(), results):
                if "macro" not in module_name.lower():
                    self.metadata[module_name] = result
                    if self.debug:
                        console.print(f"Processed: {module_name}")
            
            if self.debug:
                console.print(Panel(f"Extracted metadata for {len(self.metadata)} modules", title="[bold blue]Metadata Extraction Complete[/bold blue]", expand=False))

        async def process_module(self, module_name: str, metadata: Dict[str, Any]):
            variable_identifier = await self.create_subagent(
            VariableIdentificationAgent,
            self.parser,
            module_name,
            metadata,
            self.model,
            self.debug
        )
            #variable_identifier = VariableIdentificationAgent(self.parser, module_name, metadata, model=self.model, debug=self.debug)
            return await variable_identifier.identify_variables()

class VariableIdentificationAgent(BaseAgent):
    def __init__(self, parser: SASParser, module_name: str, metadata: Dict[str, Any], model: str = "gpt-4", debug: bool = False):
        super().__init__("VariableIdentifier", create_subagents=False, model=model, debug=debug)
        self.parser = parser
        self.module_name = module_name
        self.metadata = metadata

    async def identify_variables(self) -> Dict[str, Any]:
        context = self._prepare_context()
        prompt = self._create_prompt(context)
        response = await self.generate_response(prompt, guided_json=True, task_description=f"Identifying variables for {self.module_name}")
        variables_info = self._parse_response(response)
        
        if self.debug:
            initial_vars = set(self.metadata['metadata']['global'] + self.metadata['metadata']['local'] + self.metadata['metadata']['used'])
            llm_vars = set(variables_info['variables'])
            new_vars = llm_vars - initial_vars
            missing_vars = initial_vars - llm_vars
            
            if new_vars or missing_vars:
                table = Table(title=f"Variable Comparison for {self.module_name}")
                table.add_column("New Variables", style="green")
                table.add_column("Missing Variables", style="red")
                for new, missing in zip_longest(sorted(new_vars), sorted(missing_vars), fillvalue=""):
                    table.add_row(new, missing)
                console.print(table)
        
        return self.parser.get_module_comprehensive_metadata(self.module_name, variables_info)

    def _prepare_context(self) -> str:
        context = f"########### Main Module ({self.module_name}): ############\n{self.parser.sas_modules[self.module_name]}\n\n"
        for ext_file, ext_metadata in self.metadata['metadata']['external_files'].items():
            if ext_file in self.parser.sas_modules:
                context += f"########### External File ({ext_file}):############\n{self.parser.sas_modules[ext_file]}\n\n"
        return context

    def _create_prompt(self, context: str) -> str:
        return f"""
        Analyze the following SAS code and identify all variables used in the main module and its referenced external files.
        Provide the output in JSON format as follows:
        {{
            "variables": [list of all variables in the main module],
            "external": {{
                "filename1": [list of variables in this external file],
                "filename2": [list of variables in this external file],
                ...
            }}
        }}

        Context:
        {context}

        Please provide your analysis in the specified JSON format.
        """

    def _parse_response(self, response: str) -> Dict[str, Any]:
        # Assuming the response is already in JSON format due to guided_json=True
        return response

class ConversionAgent(BaseAgent):
    def __init__(self, modules: Dict[str, str], metadata: Dict[str, Any], model: str = "gpt-4", debug: bool = False):
        super().__init__("Conversion Manager", create_subagents=True, model=model, debug=debug)
        self.modules = modules
        self.metadata = metadata
        self.converted_modules = {}

    async def run_conversion(self) -> Dict[str, Dict[str, str]]:
        tasks = []
        for module_name, module_metadata in self.metadata.items():
            if "macro" not in module_name.lower():
                task = self._convert_module(module_name, module_metadata)
                tasks.append(task)

        results = await asyncio.gather(*tasks)
        for module_name, converted_code in results:
            self.converted_modules[module_name] = converted_code

        return self.converted_modules

    async def _convert_module(self, module_name: str, module_metadata: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        module_code = self.modules[module_name]
        macro_codes = self._get_macro_codes(module_metadata)
        
        conversion_subagent = await self.create_subagent(
            CodeConversionSubAgent,
            module_name,
            module_code,
            macro_codes,
            module_metadata,
            self.model,
            self.debug
        )

        converted_code = await conversion_subagent.convert()
        return module_name, converted_code

    def _get_macro_codes(self, module_metadata: Dict[str, Any]) -> Dict[str, str]:
        macro_codes = {}
        for macro_name in module_metadata['metadata']['external_files']:
            if macro_name in self.modules:
                macro_codes[macro_name] = self.modules[macro_name]
        return macro_codes

class CodeConversionSubAgent(BaseAgent):
    def __init__(self, module_name: str, module_code: str, macro_codes: Dict[str, str], metadata: Dict[str, Any], model: str = "gpt-4", debug: bool = False):
        super().__init__(f"Conversion Sub-Agent: {module_name}", create_subagents=True, model=model, debug=debug)
        self.module_name = module_name
        self.module_code = module_code
        self.macro_codes = macro_codes
        self.metadata = metadata
        self.reflector = None

    async def convert(self) -> Dict[str, str]:
        converted_code = await self._initial_conversion()
        self.reflector = await self.create_subagent(
            ReflectorAgent,
            self.module_name,
            self.model,
            self.debug
        )

        iteration = 0
        while True:
            if iteration == 0:
                reflection = await self.reflector.reflect(self.get_history(), self.module_code, self.macro_codes, converted_code, self.metadata)
            else:
                reflection = await self.reflector.reflect_on_update(self.get_last_message()['content'])

            if self.debug:
                console.print(Panel(f"Iteration {iteration + 1}\nPass: {reflection['pass']}\nSuggestions: {reflection['suggestions']}", 
                                    title=f"[bold blue]{self.name} Reflection[/bold blue]"))
            
            if reflection['pass']:
                break
            
            converted_code = await self._improve_code(converted_code, reflection['suggestions'])
            iteration += 1

        return {
            'module': converted_code['module'],
            **{macro: converted_code[macro] for macro in self.macro_codes}
        }

    async def _initial_conversion(self) -> Dict[str, str]:
        prompt = self._create_conversion_prompt()
        response = await self.generate_response(prompt, guided_json=True,task_description=f"Converting {self.module_name}")
        return self._parse_conversion_response(response)

    def _create_conversion_prompt(self) -> str:
        prompt = f"Convert the following SAS module and its macros to Python:\n\nMain Module ({self.module_name}):\n{self.module_code}\n\n"
        for macro_name, macro_code in self.macro_codes.items():
            prompt += f"Macro ({macro_name}):\n{macro_code}\n\n"
        prompt += f"Metadata:\n{self.metadata}\n\nProvide the converted Python code for the main module and each macro separately.\n\n"
        prompt += """
                    Provide the output in JSON format as follows:        
                    {
                            'code':python code,
                            'external':{"filename1.py":python code,
                                        "filename2.py":python code
                                        }
                    }
                  """

        return prompt

    def _parse_conversion_response(self, response: str) -> Dict[str, str]:
        # Implement parsing logic to extract converted code for module and macros
        # This is a placeholder implementation
        return {f'{self.module_name}.py': response} #, **{macro: response for macro in self.macro_codes}

    async def _improve_code(self, code: Dict[str, str], suggestions: List[str]) -> Dict[str, str]:
        prompt = f"Improve the following Python code based on these suggestions:\n\nCode:\n{code}\n\nSuggestions:\n" + "\n".join(suggestions)
        response = await self.generate_response(prompt,guided_json=True, task_description=f"Improving {self.module_name}")
        return self._parse_conversion_response(response)

class ReflectorAgent(BaseAgent):
    def __init__(self, module_name: str, model: str = "gpt-4", debug: bool = False):
        super().__init__(f"Reflector: {module_name}", create_subagents=False, model=model, debug=debug)
        self.module_name = module_name

    async def reflect(self, history: List[Dict[str, str]], original_module: str, original_macros: Dict[str, str], converted_code: Dict[str, str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._create_initial_reflection_prompt(history, original_module, original_macros, converted_code, metadata)
        response = await self.generate_response(prompt, task_description=f"Initial reflection on {self.module_name}")
        return self._parse_reflection_response(response)

    async def reflect_on_update(self, last_message: str) -> Dict[str, Any]:
        prompt = self._create_update_reflection_prompt(last_message)
        response = await self.generate_response(prompt, task_description=f"Reflecting on update for {self.module_name}")
        return self._parse_reflection_response(response)

    def _create_initial_reflection_prompt(self, history: List[Dict[str, str]], original_module: str, original_macros: Dict[str, str], converted_code: Dict[str, str], metadata: Dict[str, Any]) -> str:
        prompt = f"""
        Analyze the following SAS to Python conversion process:

        Conversion History:
        {self._format_history(history)}

        Original SAS Module:
        {original_module}

        Original SAS Macros:
        {original_macros}

        Latest Converted Python Code:
        {converted_code}

        Metadata:
        {metadata}

        Your task is to:
        1. Determine if the conversion is satisfactory (pass) or needs improvement (fail).
        2. If improvement is needed, provide specific suggestions to improve the code, change something, or add functionality.
        3. Ensure all SAS functionality has been properly translated.
        4. Verify that all variables from the metadata are accounted for.
        5. Check that Python best practices and error handling are followed.
        6. Consider the full conversion history and requirements when making your assessment.

        Provide your reflection in the following format:
        PASS: [True/False]
        SUGGESTIONS:
        - [Suggestion 1]
        - [Suggestion 2]
        ...
        EXPLANATION: [Brief explanation of your decision and suggestions]
        """
        return prompt

    def _create_update_reflection_prompt(self, last_message: str) -> str:
        prompt = f"""
        Analyze the following update to the Python conversion:

        Latest Update:
        {last_message}

        Based on the previous recommendations, new code was generated. Your task is to:
        1. Determine if the conversion is now satisfactory (pass) or still needs improvement (fail).
        2. If improvement is still needed, provide specific suggestions to improve the code, change something, or add functionality.
        3. Ensure all previously identified issues have been addressed.
        4. Check that the new changes haven't introduced any new problems.

        Provide your reflection in the following format:
        PASS: [True/False]
        SUGGESTIONS:
        - [Suggestion 1]
        - [Suggestion 2]
        ...
        EXPLANATION: [Brief explanation of your decision and suggestions]
        """
        return prompt

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        formatted_history = ""
        for entry in history:
            formatted_history += f"{entry['role'].capitalize()}: {entry['content']}\n\n"
        return formatted_history

    def _parse_reflection_response(self, response: str) -> Dict[str, Any]:
        lines = response.split('\n')
        result = {
            'pass': False,
            'suggestions': [],
            'explanation': ''
        }
        
        for line in lines:
            if line.startswith('PASS:'):
                result['pass'] = line.split(':')[1].strip().lower() == 'true'
            elif line.startswith('SUGGESTIONS:'):
                continue
            elif line.startswith('-'):
                result['suggestions'].append(line[1:].strip())
            elif line.startswith('EXPLANATION:'):
                result['explanation'] = line.split(':')[1].strip()

        return result

# # Usage example:
# sas_modules = {...}  # Your SAS modules dictionary
# metadata_agent = MetadataExtractionAgent(sas_modules, debug=True)
# updated_metadata = metadata_agent.metadata

import os
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from RobustCodeInterpreter import RobustCodeInterpreter
import os
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from RobustCodeInterpreter import RobustCodeInterpreter

console = Console()

class ExecutionAgent(BaseAgent):
    def __init__(self, working_folder: str, max_fix_iterations: int = 3, model: str = "gpt-4", debug: bool = False):
        super().__init__("ExecutionAgent", create_subagents=False, model=model, debug=debug)
        self.working_folder = working_folder
        self.max_fix_iterations = max_fix_iterations
        self.interpreter = RobustCodeInterpreter(working_folder)

    def execute(self, module_name: str, python_code: Dict[str, Any], sas_code: Dict[str, str], metadata: Dict[str, Any]):
        if self.debug:
            console.print(Panel(f"Starting execution process for module: {module_name}", title="[bold cyan]ExecutionAgent[/bold cyan]"))

        self.interpreter.add_module(module_name, python_code['code'])
        for macro_name, macro_code in python_code.get('external', {}).items():
            self.interpreter.add_module(macro_name, macro_code)

        result = self.execute_module(module_name, python_code, sas_code, metadata)

        if result["success"]:
            self.save_output(module_name, python_code['code'])
            for macro_name, macro_code in python_code.get('external', {}).items():
                self.save_output(macro_name, macro_code)
        else:
            if self.debug:
                console.print(f"[bold red]Execution failed for module: {module_name}[/bold red]")

        self.interpreter.reset()
        return result

    def execute_module(self, module_name: str, python_code: Dict[str, Any], sas_code: Dict[str, str], metadata: Dict[str, Any]):
        fixer = CodeFixerSubagent(self.model, self.debug)

        for attempt in range(self.max_fix_iterations):
            if self.debug:
                console.print(f"[cyan]Execution attempt {attempt + 1} for module: {module_name}[/cyan]")

            result = self.interpreter.execute_module(module_name)

            if result["success"]:
                if self.debug:
                    console.print(f"[green]Module {module_name} executed successfully[/green]")
                return result

            if self.debug:
                console.print(f"[yellow]Execution failed. Attempting to fix. Error: {result['error']}[/yellow]")

            fixed_code = fixer.fix_code(
                module_name,
                sas_code,
                python_code['code'],
                python_code.get('external', {}),
                metadata,
                result["error"],
                attempt
            )

            if fixed_code == python_code['code']:
                if self.debug:
                    console.print("[bold red]Code fixer couldn't improve the code. Stopping attempts.[/bold red]")
                return result

            python_code['code'] = fixed_code
            self.interpreter.add_module(module_name, fixed_code)

        if self.debug:
            console.print(f"[bold red]Failed to fix module {module_name} after {self.max_fix_iterations} attempts[/bold red]")
        return result

    def save_output(self, module_name: str, code: str):
        output_path = os.path.join(self.working_folder, module_name)
        with open(output_path, 'w') as f:
            f.write(code)
        if self.debug:
            console.print(f"[green]Output for {module_name} saved to {output_path}[/green]")
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel

console = Console()

class CodeFixerSubagent(BaseAgent):
    def __init__(self, model: str = "gpt-4", debug: bool = False):
        super().__init__("CodeFixerSubagent", create_subagents=False, model=model, debug=debug)
        self.initial_prompt = None

    def fix_code(self, module_name: str, sas_code: Dict[str, str], python_code: str, 
                 python_macros: Dict[str, str], metadata: Dict[str, Any], 
                 error: str, attempt: int) -> str:
        if self.debug:
            console.print(Panel(f"Attempting to fix code for module: {module_name}", title="[bold yellow]CodeFixerSubagent[/bold yellow]"))

        if attempt == 0:
            self.initial_prompt = self._create_initial_prompt(module_name, sas_code, python_code, python_macros, metadata)
            prompt = self.initial_prompt + f"\n\nError message:\n{error}\n\nPlease provide the corrected Python code for the module."
        else:
            prompt = f"""
            Previous attempt to fix the code was unsuccessful. Here's the current state of the Python module:

            ```python
            {self._add_line_numbers(python_code)}
            ```

            Error message:
            {error}

            Please provide an updated version of the Python code that addresses this error.
            """

        response = self.generate_response(prompt, task_description=f"Fixing code for {module_name}")
        
        fixed_code = self._parse_fix_response(response)

        if self.debug:
            console.print(Panel(fixed_code, title="[bold green]Fixed Code[/bold green]", expand=False))

        return fixed_code

    def _create_initial_prompt(self, module_name: str, sas_code: Dict[str, str], 
                               python_code: str, python_macros: Dict[str, str], 
                               metadata: Dict[str, Any]) -> str:
        prompt = f"""
        Fix the following Python code based on the original SAS code and the metadata:

        Module name: {module_name}

        Original SAS module:
        ```sas
        {self._add_line_numbers(sas_code[module_name])}
        ```

        Python module to be fixed:
        ```python
        {self._add_line_numbers(python_code)}
        ```

        SAS Macros:
        {self._format_macros(sas_code, 'sas')}

        Python Macros:
        {self._format_macros(python_macros, 'python')}

        Metadata:
        {metadata}
        """
        return prompt

    def _add_line_numbers(self, code: str) -> str:
        lines = code.split('\n')
        return '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))

    def _format_macros(self, macros: Dict[str, str], language: str) -> str:
        formatted = ""
        for macro_name, macro_code in macros.items():
            if macro_name.lower() != 'code':  # Skip the main module code
                formatted += f"\n{macro_name}:\n```{language}\n{self._add_line_numbers(macro_code)}\n```\n"
        return formatted

    def _parse_fix_response(self, response: str) -> str:
        start = response.find("```python")
        end = response.rfind("```")
        if start != -1 and end != -1:
            return response[start+9:end].strip()
        return response.strip()
# class ExecutionAgent(BaseAgent):
#     def __init__(self, working_folder: str, max_fix_iterations: int = 3, model: str = "gpt-4", debug: bool = False):
#         super().__init__("ExecutionAgent", create_subagents=False, model=model, debug=debug)
#         self.working_folder = working_folder
#         self.max_fix_iterations = max_fix_iterations
#         self.interpreter = RobustCodeInterpreter(working_folder)

#     def execute(self, python_codes: Dict[str, Dict[str, Any]], sas_codes: Dict[str, str], metadata: Dict[str, Dict[str, Any]]):
#         if self.debug:
#             console.print(Panel("Starting execution process", title="[bold cyan]ExecutionAgent[/bold cyan]"))

#         results = {}
#         for module_name, module_data in python_codes.items():
#             self.interpreter.add_module(module_name, module_data['code'])
#             for ext_name, ext_code in module_data.get('external', {}).items():
#                 self.interpreter.add_module(ext_name, ext_code)

#             result = self.execute_module(module_name, sas_codes, python_codes, metadata)
#             results[module_name] = result

#             if result["success"]:
#                 self.save_output(module_name, module_data['code'])
#             else:
#                 if self.debug:
#                     console.print(f"[bold red]Execution failed for module: {module_name}[/bold red]")

#             self.interpreter.reset()

#         return results

#     def execute_module(self, module_name: str, sas_codes: Dict[str, str], python_codes: Dict[str, Dict[str, Any]], metadata: Dict[str, Dict[str, Any]]):
#         module_metadata = metadata.get(module_name, {})
#         module_data = python_codes[module_name]

#         fixer = CodeFixerSubagent(self.model, self.debug)

#         for attempt in range(self.max_fix_iterations):
#             if self.debug:
#                 console.print(f"[cyan]Execution attempt {attempt + 1} for module: {module_name}[/cyan]")

#             result = self.interpreter.execute_module(module_name)

#             if result["success"]:
#                 if self.debug:
#                     console.print(f"[green]Module {module_name} executed successfully[/green]")
#                 return result

#             if self.debug:
#                 console.print(f"[yellow]Execution failed. Attempting to fix. Error: {result['error']}[/yellow]")

#             fixed_code = fixer.fix_code(
#                 module_name,
#                 sas_codes[module_name],
#                 module_data['code'],
#                 sas_codes,
#                 {name: code for name, code in module_data['external'].items()},
#                 module_metadata,
#                 result["error"],
#                 attempt
#             )

#             if fixed_code == module_data['code']:
#                 if self.debug:
#                     console.print("[bold red]Code fixer couldn't improve the code. Stopping attempts.[/bold red]")
#                 return result

#             module_data['code'] = fixed_code
#             self.interpreter.add_module(module_name, fixed_code)

#         if self.debug:
#             console.print(f"[bold red]Failed to fix module {module_name} after {self.max_fix_iterations} attempts[/bold red]")
#         return result

#     def save_output(self, module_name: str, code: str):
#         output_path = os.path.join(self.working_folder, module_name)
#         with open(output_path, "w") as f:
#             f.write(code)
#         if self.debug:
#             console.print(f"[green]Output for module {module_name} saved to {output_path}[/green]")

# class CodeFixerSubagent(BaseAgent):
#     def __init__(self, model: str = "gpt-4", debug: bool = False):
#         super().__init__("CodeFixerSubagent", create_subagents=False, model=model, debug=debug)
#         self.initial_prompt = None

#     def fix_code(self, module_name: str, sas_module: str, python_module: str, 
#                  sas_codes: Dict[str, str], python_macros: Dict[str, str], 
#                  metadata: Dict[str, Any], error: str, attempt: int) -> str:
#         if self.debug:
#             console.print(Panel(f"Attempting to fix code for module: {module_name}", title="[bold yellow]CodeFixerSubagent[/bold yellow]"))

#         if attempt == 0:
#             self.initial_prompt = self._create_initial_prompt(module_name, sas_module, python_module, sas_codes, python_macros, metadata)
#             prompt = self.initial_prompt + f"\n\nError message:\n{error}\n\nPlease provide the corrected Python code for the module."
#         else:
#             prompt = f"""
#             Previous attempt to fix the code was unsuccessful. Here's the current state of the Python module:

#             ```python
#             {self._add_line_numbers(python_module)}
#             ```

#             Error message:
#             {error}

#             Please provide an updated version of the Python code that addresses this error.
#             """

#         response = self.generate_response(prompt, task_description=f"Fixing code for {module_name}")
        
#         fixed_code = self._parse_fix_response(response)

#         if self.debug:
#             console.print(Panel(fixed_code, title="[bold green]Fixed Code[/bold green]", expand=False))

#         return fixed_code

#     def _create_initial_prompt(self, module_name: str, sas_module: str, python_module: str, 
#                                sas_codes: Dict[str, str], python_macros: Dict[str, str], 
#                                metadata: Dict[str, Any]) -> str:
#         prompt = f"""
#         Fix the following Python code based on the original SAS code and the metadata:

#         Module name: {module_name}

#         Original SAS module:
#         ```sas
#         {sas_module}
#         ```

#         Python module to be fixed:
#         ```python
#         {self._add_line_numbers(python_module)}
#         ```

#         SAS Macros:
#         {self._format_macros(sas_codes, 'sas')}

#         Python Macros:
#         {self._format_macros(python_macros, 'python')}

#         Metadata:
#         {metadata}
#         """
#         return prompt

#     def _add_line_numbers(self, code: str) -> str:
#         lines = code.split('\n')
#         return '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))

#     def _format_macros(self, macros: Dict[str, str], language: str) -> str:
#         formatted = ""
#         for macro_name, macro_code in macros.items():
#             formatted += f"\n{macro_name}:\n```{language}\n{self._add_line_numbers(macro_code)}\n```\n"
#         return formatted

#     def _parse_fix_response(self, response: str) -> str:
#         start = response.find("```python")
#         end = response.rfind("```")
#         if start != -1 and end != -1:
#             return response[start+9:end].strip()
#         return response.strip()

# # Usage example:
# def main():
#     working_folder = "test"
#     agent = ExecutionAgent(working_folder, debug=True)
    
#     python_codes = {
#         "module1.py": {
#             "code": "print('Hello from module 1')",
#             "external": {
#                 "Macro1.py": "def macro1(): print('Macro 1')"
#             }
#         },
#         "module2.py": {
#             "code": "print('Hello from module 2')\nprint(result_from_module1)",
#             "external": {
#                 "Macro2.py": "def macro2(): print('Macro 2')"
#             }
#         }
#     }
#     sas_codes = {
#         "module1.sas": "%macro module1; %put Hello from SAS module 1; %mend;",
#         "module2.sas": "%macro module2; %put Hello from SAS module 2; %mend;",
#         "Macro1.sas": "%macro Macro1; %put Macro 1; %mend;",
#         "Macro2.sas": "%macro Macro2; %put Macro 2; %mend;"
#     }
#     metadata = {
#         "module1.sas": {
#             "module_name": "module1.sas",
#             "metadata": {
#                 "global": ["global_var1"],
#                 "local": [],
#                 "used": [],
#                 "imports": {"global": {}, "local_Used": {}},
#                 "external_files": {"Macro1.sas": {"global": [], "local": [], "used": []}},
#                 "proc_statements": [],
#                 "data_steps": ["_null_"],
#                 "library_references": []
#             },
#             "warnings": []
#         },
#         "module2.sas": {
#             "module_name": "module2.sas",
#             "metadata": {
#                 "global": [],
#                 "local": [],
#                 "used": ["result_from_module1"],
#                 "imports": {"global": {"module1.sas": ["global_var1"]}, "local_Used": {}},
#                 "external_files": {"Macro2.sas": {"global": [], "local": [], "used": []}},
#                 "proc_statements": [],
#                 "data_steps": [],
#                 "library_references": ["module1"]
#             },
#             "warnings": []
#         }
#     }

#     results = agent.execute(python_codes, sas_codes, metadata)
#     console.print(results)

# if __name__ == "__main__":
#     main()