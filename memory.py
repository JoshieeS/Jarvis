# memory_manager.py

import json, re, os

class MemoryManager:
    def __init__(self, filename: str = "memory.json"):
        self.filename = filename
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.memory = json.load(f)
        else:
            self.memory = {}

    def update_from_input(self, text: str):
        """
        Look for patterns like "my X is Y" and store them.
        """
        # match "my <key> is <value>"
        matches = re.findall(
            r"\bmy ([\w\s]+?) is ([\w\s]+?)(?:[\.!?]|$)",
            text,
            re.IGNORECASE,
        )
        for raw_key, raw_val in matches:
            key = raw_key.strip().lower()
            val = raw_val.strip().rstrip(".!?")
            self.memory[key] = val

        # also match "hello my name is X"
        m = re.search(r"\bhello my name is ([\w\s]+)", text, re.IGNORECASE)
        if m:
            self.memory["name"] = m.group(1).strip()

        self._save()

    def summary(self) -> str:
        """
        Return a human-readable summary of all stored facts.
        """
        if not self.memory:
            return ""
        return "; ".join(f"{k.capitalize()}: {v}" for k, v in self.memory.items())

    def _save(self):
        with open(self.filename, "w") as f:
            json.dump(self.memory, f, indent=2)
