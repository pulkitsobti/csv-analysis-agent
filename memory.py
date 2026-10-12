"""Persistent memory: small facts the agent keeps between sessions.

Memories are stored in a JSON file. Each memory is a short, named fact such as
a metric definition ("revenue = Units x Unit_Price"), a correction, a reusable
filter ("good attendance = Attendance >= 90") or a preference. Raw chats and
data are never stored.

Rules:
- Scope: a "dataset" memory only applies to datasets with the same columns;
  a "global" memory applies everywhere.
- Conflicts: saving under an existing name replaces the old fact, and the old
  version is kept in the memory's history. A dataset memory overrides a global
  memory with the same name.
- Staleness: memories not updated for STALE_AFTER_DAYS are flagged, so the
  agent re-checks them against the data before relying on them.
"""

import hashlib
import json
import os
from datetime import datetime

KINDS = ("definition", "correction", "filter", "preference")
STALE_AFTER_DAYS = 30


def dataset_id(columns):
    """Datasets with the same set of column names share dataset memories."""
    key = "|".join(sorted(str(c).strip().lower() for c in columns))
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def now():
    return datetime.now().isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path):
        self.path = path
        self.memories = self.load()

    # ---------- reading and writing the file ----------

    def load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            # Keep a copy of the broken file instead of silently losing it.
            os.replace(self.path, self.path + ".corrupt")
            return []

    def write(self):
        # Write to a temporary file first so a crash can't leave half a file.
        temp_path = self.path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(self.memories, file, indent=2)
        os.replace(temp_path, self.path)

    # ---------- changing memories ----------

    def save(self, name, kind, content, scope, columns):
        """Add a memory, or replace the one with the same name and scope."""
        name = name.strip().lower().replace(" ", "_")
        content = content.strip()
        if not name or not content:
            raise ValueError("A memory needs a name and some content.")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of: {', '.join(KINDS)}")
        if scope not in ("dataset", "global"):
            raise ValueError("scope must be 'dataset' or 'global'")
        if scope == "dataset" and not columns:
            raise ValueError("Load a dataset before saving a dataset memory.")

        scope_id = dataset_id(columns) if scope == "dataset" else "global"
        existing = self.find(name, scope_id)

        if existing is None:
            self.memories.append({
                "name": name,
                "kind": kind,
                "content": content,
                "scope": scope,
                "scope_id": scope_id,
                "columns": list(map(str, columns)) if scope == "dataset" else [],
                "created_at": now(),
                "updated_at": now(),
                "history": [],
            })
            self.write()
            return {"status": "saved", "name": name, "scope": scope}

        if existing["content"] == content and existing["kind"] == kind:
            return {"status": "unchanged", "name": name}

        # Conflict: the newest fact wins, the old one goes into history.
        previous = existing["content"]
        existing["history"].append({"content": previous, "replaced_at": now()})
        existing.update(kind=kind, content=content, updated_at=now())
        self.write()
        return {"status": "updated", "name": name, "previous_content": previous}

    def delete(self, name, scope_id):
        before = len(self.memories)
        self.memories = [
            m for m in self.memories
            if not (m["name"] == name and m["scope_id"] == scope_id)
        ]
        self.write()
        return len(self.memories) < before

    def forget(self, name, columns):
        """Delete a memory by name, preferring the current dataset's version."""
        name = name.strip().lower().replace(" ", "_")
        for scope_id in (dataset_id(columns), "global"):
            if self.find(name, scope_id) and self.delete(name, scope_id):
                return {"status": "deleted", "name": name}
        names = [m["name"] for m in self.relevant(columns)]
        raise ValueError(f"No memory called '{name}'. Saved memories: {names}")

    def clear(self):
        self.memories = []
        self.write()

    # ---------- looking memories up ----------

    def find(self, name, scope_id):
        for memory in self.memories:
            if memory["name"] == name and memory["scope_id"] == scope_id:
                return memory
        return None

    def relevant(self, columns):
        """Memories that apply to a dataset with these columns. A dataset
        memory overrides a global memory with the same name."""
        current = dataset_id(columns) if columns else None
        chosen = {}
        for memory in self.memories:
            if memory["scope_id"] == "global":
                chosen.setdefault(memory["name"], memory)
        for memory in self.memories:
            if memory["scope_id"] == current:
                chosen[memory["name"]] = memory
        return list(chosen.values())

    @staticmethod
    def age_days(memory):
        updated = datetime.fromisoformat(memory["updated_at"])
        return (datetime.now() - updated).days

    def is_stale(self, memory):
        return self.age_days(memory) > STALE_AFTER_DAYS

    def as_prompt_text(self, columns):
        """The relevant memories, formatted for the system prompt."""
        lines = []
        for memory in self.relevant(columns):
            line = (
                f"- {memory['name']} [{memory['kind']}, {memory['scope']}]: "
                f"{memory['content']} (updated {memory['updated_at'][:10]})"
            )
            if self.is_stale(memory):
                line += " POSSIBLY STALE: verify with tools before relying on it."
            lines.append(line)
        return "\n".join(lines) if lines else "(none yet)"
