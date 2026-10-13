"""The agent: connects Gemini to the data-analysis tools.

Gemini never sees the full dataset. It only sees the column names and
whatever the tools return, so every number in an answer has to come from
a real pandas calculation.
"""

import os

from google import genai
from google.genai import errors, types

import tools

# Models to try in order. If one is rate-limited or overloaded, the next is
# used (each model has its own free-tier quota). Override with GEMINI_MODELS.
MODELS = [
    name.strip()
    for name in os.getenv(
        "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite"
    ).split(",")
]
MAX_TOOL_CALLS = 12

SYSTEM_PROMPT = """You are a data analysis agent working on a CSV file the user uploaded.

Rules:
1. Answer by calling your tools. Never guess or do arithmetic in your head:
   every number in your answer must come from a tool result.
2. Use exact column names from the schema below. Call dataset_overview only if
   the schema is not enough.
3. To analyse part of the data, call filter_rows with save_as="some_name", then pass
   subset="some_name" to the other tools. Chain filters with the `source` argument.
4. If a metric is not a column (e.g. revenue = units x price), build it with
   add_calculated_column first.
5. When you need several independent results, request those tool calls together
   in the same step.
6. If the question cannot be answered from this dataset, say so plainly.
7. Keep answers short and clear. Mention which analysis each number came from.
"""

MEMORY_PROMPT = """
Memory (facts saved in earlier sessions are listed at the end):
- Apply saved memories when relevant, e.g. use a saved definition or filter instead
  of guessing what a term means, and say which memory you used.
- Call save_memory when the user asks you to remember something, defines a term or
  metric, corrects you, or names a filter they want to reuse. Never save one-off
  answers, raw rows or whole results.
- If a new fact conflicts with a saved memory, save it under the SAME name so it
  replaces the old one, and tell the user what changed.
- Call forget_memory when the user asks you to forget something.
"""


class DataAgent:
    def __init__(self, df, api_key, memory=None):
        self.original_df = df
        self.columns = list(df.columns)
        self.memory = memory  # a MemoryStore, or None to turn memory off
        self.reset()
        # Retry once on temporary server errors. Rate limits (429) are not
        # retried here; ask() moves on to the next model instead.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=2, initial_delay=2, http_status_codes=[500, 502, 503, 504]
                )
            ),
        )

    def reset(self):
        self.df = self.original_df
        # Named row subsets created by filter_rows. "all" is the whole dataset.
        self.subsets = {"all": self.df}

    def get_subset(self, name):
        if name not in self.subsets:
            raise ValueError(
                f"No subset called '{name}'. Available subsets: {list(self.subsets)}"
            )
        return self.subsets[name]

    def tool_list(self):
        """The functions Gemini is allowed to call.

        Gemini reads each function's name, type hints and docstring to decide
        when and how to call it. If a function raises an error, the SDK sends
        the error message back to Gemini so it can correct itself (e.g. after
        using a wrong column name).

        They are plain functions defined in here (not methods) because the SDK
        deep-copies its config, and copying a method would copy the whole agent.
        """
        agent = self

        def dataset_overview() -> dict:
            """Get the row count, each column's name, type, missing-value count and
            example values, plus a few sample rows."""
            return tools.dataset_overview(agent.df)

        def filter_rows(column: str, operator: str, value: str,
                        source: str = "all", save_as: str = "") -> dict:
            """Keep only the rows where `column` matches a condition.
            operator: equals, not_equals, contains, >, <, >=, <= (text matching is
            case-insensitive; > and < also work on dates like 2026-03-01).
            source: subset to filter, default 'all' (the whole dataset).
            save_as: name to save the result under, so other tools can analyse it
            through their `subset` argument."""
            data = agent.get_subset(source)
            result = tools.filter_rows(data, column, operator, value)
            if save_as:
                if save_as == "all":
                    raise ValueError("'all' is reserved for the whole dataset.")
                agent.subsets[save_as] = result
            return {
                "matched_rows": len(result),
                "out_of": len(data),
                "saved_as": save_as or None,
                "preview": tools.rows_to_records(result),
            }

        def column_statistics(column: str, subset: str = "all") -> dict:
            """Count, sum, mean, median, standard deviation, min, quartiles and max
            of one numeric column."""
            return tools.column_statistics(agent.get_subset(subset), column)

        def group_and_aggregate(group_by: str, value_column: str,
                                operation: str = "mean", subset: str = "all") -> dict:
            """For each value of `group_by`, compute mean, sum, count, min, max or
            median of `value_column`. Results are sorted highest first."""
            return tools.group_and_aggregate(
                agent.get_subset(subset), group_by, value_column, operation
            )

        def correlations(subset: str = "all") -> dict:
            """Correlation between every pair of numeric columns, strongest first."""
            return tools.correlations(agent.get_subset(subset))

        def top_rows(column: str, n: int = 5, largest: bool = True,
                     subset: str = "all") -> dict:
            """The n rows with the largest (largest=True) or smallest (largest=False)
            values in a numeric column. Use for 'who/which is highest/lowest' questions."""
            return tools.top_rows(agent.get_subset(subset), column, n, largest)

        def value_counts(column: str, top: int = 10, subset: str = "all") -> dict:
            """How many times each value appears in a column, with percentages.
            Use for 'how many per category' or 'most common' questions."""
            return tools.value_counts(agent.get_subset(subset), column, top)

        def find_outliers(column: str, subset: str = "all") -> dict:
            """Find unusually high or low values in a numeric column (1.5 x IQR rule)."""
            return tools.find_outliers(agent.get_subset(subset), column)

        def add_calculated_column(new_column: str, column_a: str,
                                  operation: str, column_b: str) -> dict:
            """Create a new column = column_a (add | subtract | multiply | divide) column_b.
            column_b can be a column name or a number. The new column is then
            available to every other tool and subset."""
            if new_column in agent.df.columns:
                raise ValueError(f"Column '{new_column}' already exists, use it directly.")
            for name, data in agent.subsets.items():
                new_values = tools.calculate_column(data, column_a, operation, column_b)
                agent.subsets[name] = data.assign(**{new_column: new_values})
            agent.df = agent.subsets["all"]
            return {
                "created": new_column,
                "formula": f"{column_a} {operation} {column_b}",
                "statistics": tools.column_statistics(agent.df, new_column),
            }

        def save_memory(name: str, kind: str, content: str, scope: str = "dataset") -> dict:
            """Save a fact for future sessions.
            name: short snake_case name, e.g. good_attendance.
            kind: definition, correction, filter or preference.
            content: the fact in one short sentence using exact column names,
            e.g. 'revenue = Units multiply Unit_Price' or 'Attendance >= 90'.
            scope: 'dataset' (only datasets with these columns) or 'global' (all datasets).
            Saving under an existing name replaces that memory."""
            return agent.memory.save(name, kind, content, scope, agent.columns)

        def forget_memory(name: str) -> dict:
            """Delete a saved memory by name, when the user asks you to forget it."""
            return agent.memory.forget(name, agent.columns)

        analysis_tools = [
            dataset_overview,
            filter_rows,
            column_statistics,
            group_and_aggregate,
            correlations,
            top_rows,
            value_counts,
            find_outliers,
            add_calculated_column,
        ]
        if self.memory is None:
            return analysis_tools
        return analysis_tools + [save_memory, forget_memory]

    # ---------- running a question ----------

    def system_prompt(self):
        # A compact schema (type + example values per column) means Gemini
        # rarely needs an extra request just to look at the data.
        schema = "\n".join(
            f"- {name} ({self.df[name].dtype}), e.g. {[str(v) for v in self.df[name].dropna().unique()[:3]]}"
            for name in self.df.columns
        )
        prompt = SYSTEM_PROMPT + f"\nDataset: {len(self.df)} rows.\nColumns:\n{schema}\n"
        if self.memory is not None:
            prompt += MEMORY_PROMPT + "\nSaved memories:\n" + self.memory.as_prompt_text(self.columns) + "\n"
        return prompt

    def ask(self, question, history=(), first_model=None):
        """Answer a question. `history` is a list of earlier chat messages
        ({"role": "user" | "assistant", "content": str}) for follow-up questions.
        `first_model` is tried first; the other models in MODELS are backups.

        Returns {"answer": str, "steps": [{"tool", "args", "result"}, ...], "model": str}.
        """
        earlier = []
        for message in list(history)[-6:]:
            role = "user" if message["role"] == "user" else "model"
            earlier.append(types.Content(role=role, parts=[types.Part(text=message["content"])]))

        models = list(MODELS)
        if first_model:
            models = [first_model] + [m for m in MODELS if m != first_model]

        for model in models:
            self.reset()  # undo anything a failed attempt changed
            chat = self.client.chats.create(
                model=model,
                history=earlier,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt(),
                    tools=self.tool_list(),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=MAX_TOOL_CALLS
                    ),
                ),
            )
            try:
                # The SDK runs the tool loop for us: Gemini asks for a tool, the
                # SDK calls our Python function, sends the result back, and
                # repeats until Gemini writes a final text answer.
                response = chat.send_message(question)
                break
            except errors.APIError as error:
                rate_limited_or_busy = error.code in (429, 500, 503, 504)
                if not rate_limited_or_busy or model == models[-1]:
                    raise

        answer = response.text
        if not answer:
            answer = (
                "I couldn't finish this analysis within the step limit. "
                "Try asking a more specific question."
            )
        return {"answer": answer, "steps": self.read_steps(chat.get_history()), "model": model}

    @staticmethod
    def read_steps(history):
        """Pull every tool call (and its result) out of the chat history the
        SDK recorded, so the app can show how the answer was made."""
        steps = []
        for content in history:
            for part in content.parts or []:
                if part.function_call:
                    steps.append({
                        "tool": part.function_call.name,
                        "args": dict(part.function_call.args or {}),
                        "result": None,
                    })
                elif part.function_response:
                    # Match the response to the earliest unanswered call with that name.
                    for step in steps:
                        if step["tool"] == part.function_response.name and step["result"] is None:
                            step["result"] = part.function_response.response
                            break
        return steps
