# src/database.py
import os
import json
from typing import List, Dict, Any

class TrialDatabase:
    """Handles persistent storage and keyword searching for clinical trials."""
    def __init__(self, storage_path: str = "data/indexed_trials.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            self._save_database([])

    def _load_database(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_database(self, database: List[Dict[str, Any]]):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(database, f, indent=2, ensure_ascii=False)

    def add_trial(self, trial_id: str, title: str, structured_criteria: Dict[str, List[str]]):
        db = self._load_database()
        trial_record = {
            "trial_id": trial_id,
            "title": title,
            "inclusion_criteria": structured_criteria.get("inclusion_criteria", []),
            "exclusion_criteria": structured_criteria.get("exclusion_criteria", [])
        }
        db = [t for t in db if t["trial_id"] != trial_id]
        db.append(trial_record)
        self._save_database(db)

    def get_trial(self, trial_id: str) -> Dict[str, Any]:
        db = self._load_database()
        return next((t for t in db if t["trial_id"] == trial_id), None)

    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        db = self._load_database()
        results = []
        for trial in db:
            all_rules = trial["inclusion_criteria"] + trial["exclusion_criteria"]
            matches = [rule for rule in all_rules if keyword.lower() in rule.lower()]
            if matches:
                results.append({"trial_id": trial["trial_id"], "title": trial["title"], "matching_rules": matches})
        return results
