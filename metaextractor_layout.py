import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress
from typing import Optional

console = Console()

class BaseAgent:
    def __init__(self, name: str, model: str = "gpt-4", debug: bool = False):
        self.name = name
        self.model = model
        self.debug = debug
        self.logger = self._setup_logger()
        self.live: Optional[Live] = None
        self.layout: Optional[Layout] = None
        self.progress: Optional[Progress] = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        if not logger.handlers:
            handler = RichHandler(console=console, show_time=False, show_path=False)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        return logger

    def setup_live_display(self):
        if not self.live:
            self.layout = Layout()
            self.layout.split(
                Layout(name="top", ratio=4),
                Layout(name="bottom", ratio=1)
            )
            self.layout["top"].split_row(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=2)
            )
            self.layout["left"].split(
                Layout(name="progress"),
                Layout(name="status")
            )
            self.layout["bottom"].update(Panel("", title="Logs"))
            self.progress = Progress()
            self.layout["progress"].update(self.progress)
            self.live = Live(self.layout, console=console, refresh_per_second=4)
            self.live.start()

    def log(self, message: str, level: str = "info"):
        getattr(self.logger, level)(message)
        if self.live and self.layout:
            logs = self.layout["bottom"].renderable
            content = logs.renderable + "\n" + message
            self.layout["bottom"].update(Panel(content, title="Logs"))
            self.live.refresh()

    def update_status(self, content):
        if self.live and self.layout:
            self.layout["status"].update(content)
            self.live.refresh()

    def update_main_content(self, content):
        if self.live and self.layout:
            self.layout["right"].update(content)
            self.live.refresh()

    def cleanup_live_display(self):
        if self.live:
            self.live.stop()
            self.live = None
            self.layout = None
            self.progress = None


import asyncio
from typing import Dict, Any
from rich.table import Table
from rich.panel import Panel

class MetadataExtractionAgent(BaseAgent):
    def __init__(self, sas_modules: Dict[str, str], model: str = "gpt-4", debug: bool = False):
        super().__init__("MetadataExtractor", model=model, debug=debug)
        self.sas_modules = sas_modules
        self.parser = SASParser(sas_modules)
        self.metadata: Dict[str, Any] = {}

    async def run_extraction(self) -> Dict[str, Any]:
        self.setup_live_display()
        try:
            self.log("Starting metadata extraction process")
            
            extraction_task = self.progress.add_task("Extracting metadata", total=len(self.sas_modules))
            
            for module_name, metadata in self.parser.meta_data.items():
                if "macro" not in module_name.lower():
                    result = await self.process_module(module_name, metadata)
                    self.metadata[module_name] = result
                    self.log(f"Processed: {module_name}")
                    self.progress.update(extraction_task, advance=1)
                    
                    # Update the main content with a summary table
                    table = Table(title="Extracted Metadata Summary")
                    table.add_column("Module")
                    table.add_column("Variables")
                    for mod, data in self.metadata.items():
                        table.add_row(mod, ", ".join(data.get('variables', [])))
                    self.update_main_content(Panel(table))
            
            self.log(f"Extracted metadata for {len(self.metadata)} modules")
            return self.metadata
        finally:
            self.cleanup_live_display()

    async def process_module(self, module_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        variable_identifier = VariableIdentificationAgent(self.parser, module_name, metadata, self.model, self.debug)
        variable_identifier.live = self.live
        variable_identifier.layout = self.layout
        variable_identifier.progress = self.progress
        return await variable_identifier.identify_variables()

    # ... (other methods remain the same)

from typing import Dict, Any
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import json
from itertools import zip_longest

class VariableIdentificationAgent(BaseAgent):
    def __init__(self, parser: SASParser, module_name: str, metadata: Dict[str, Any], model: str = "gpt-4", debug: bool = False):
        super().__init__("VariableIdentifier", model=model, debug=debug)
        self.parser = parser
        self.module_name = module_name
        self.metadata = metadata

    async def identify_variables(self) -> Dict[str, Any]:
        self.log(f"Identifying variables for module: {self.module_name}")
        context = self._prepare_context()
        prompt = self._create_prompt(context)
        response = await self.generate_response(prompt, guided_json=True, task_description=f"Identifying variables for {self.module_name}")
        variables_info = self._parse_response(response)
        
        self._display_prompt_and_response(prompt, response)
        self._display_variable_comparison(variables_info)
        
        return self.parser.get_module_comprehensive_metadata(self.module_name, variables_info)

    def _display_prompt_and_response(self, prompt: str, response: str) -> None:
        prompt_panel = Panel(Syntax(prompt, "text", theme="monokai"), title="Prompt", expand=False)
        response_panel = Panel(Syntax(json.dumps(json.loads(response), indent=2), "json", theme="monokai"), title="Response", expand=False)
        
        self.update_main_content(
            Panel(
                prompt_panel + "\n" + response_panel,
                title=f"Prompt and Response for {self.module_name}"
            )
        )

    def _display_variable_comparison(self, variables_info: Dict[str, Any]) -> None:
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
            
            self.update_status(Panel(table))

    # ... (other methods remain the same)