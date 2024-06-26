import json
import os
from typing import Dict, Any

class StateManager:
    def __init__(self, state_file: str = "conversion_state.json"):
        self.state_file = state_file
        self.current_state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "input_processed": False,
            "metadata_extracted": {},
            "code_converted": {},
            "code_executed": {},
            "output_generated": False
        }

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.current_state, f)

    def update_state(self, key: str, value: Any):
        self.current_state[key] = value
        self.save_state()

    def get_state(self, key: str) -> Any:
        return self.current_state.get(key)

    def is_completed(self, step: str) -> bool:
        if step == "input_processed":
            return self.current_state["input_processed"]
        elif step in ["metadata_extracted", "code_converted", "code_executed"]:
            return all(self.current_state[step].values())
        elif step == "output_generated":
            return self.current_state["output_generated"]
        return False

    def mark_module_completed(self, step: str, module: str):
        if step in ["metadata_extracted", "code_converted", "code_executed"]:
            self.current_state[step][module] = True
            self.save_state()

# Usage example:
# state_manager = StateManager()
# if not state_manager.is_completed("input_processed"):
#     # Process input
#     state_manager.update_state("input_processed", True)
# 
# for module in modules:
#     if not state_manager.is_completed("metadata_extracted", module):
#         # Extract metadata for module
#         state_manager.mark_module_completed("metadata_extracted", module)
#     # ... continue with other steps