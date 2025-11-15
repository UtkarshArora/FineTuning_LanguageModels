import os
import re
import json

# def read_schema(schema_path):
#     """
#     Read the .schema file
#     """
#     if not os.path.exists(schema_path):
#         raise FileNotFoundError(f"Schema file not found at: {schema_path}")


#     with open(schema_path, "r") as f:
#         schema = f.read()
#     return schema
def read_schema(schema_path):
    """Read and format the schema file"""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")

    with open(schema_path, "r") as f:
        schema_data = json.load(f)

    # Format nicely for the LLM
    tables = schema_data.get("ents", {})
    main_tables = ["flight", "airline", "airport", "fare", "city", "aircraft"]

    formatted_schema = "Database Tables:\n"
    for table_name in main_tables:
        if table_name in tables:
            columns = ", ".join(list(tables[table_name].keys())[:8])  # Limit columns
            formatted_schema += f"- {table_name}: {columns}\n"

    return formatted_schema


# def extract_sql_query(response):
#     """
#     Extract the SQL query from the model's response
#     """
#     match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()

#     # Pattern 2: Look for ``` ... ``` (generic markdown block)
#     match = re.search(r"```\n(.*?)\n```", response, re.DOTALL)
#     if match:
#         return match.group(1).strip()

#     # Pattern 3: Look for the first SELECT statement, as a fallback
#     # This is less robust but can catch queries without markdown
#     match = re.search(r"SELECT .*?;", response, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(0).strip()

#     # Pattern 4: If no markdown and no semi-colon, find the full SELECT statement
#     match = re.search(r"SELECT .*$", response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
#     if match:
#         return match.group(0).strip()

#     # If no query is found, return the most likely "answer" part
#     # For Gemma-it, the response starts with 'sure, here is the query:\n\n'
#     # and the answer follows.
#     # We can try to split by 'model\n' and take the last part.
#     parts = response.split("model\n")
#     if len(parts) > 1:
#         potential_query = parts[-1].strip()
#         # If it looks like SQL, return it
#         if potential_query.lower().startswith("select"):
#             return potential_query

#     # If all else fails, return an empty string or a placeholder
#     # This will be marked as an error by the evaluator
#     print(
#         f"--- WARNING: Could not extract SQL from response: ---\n{response}\n"
#         + "-" * 50
#     )
#     return ""  # Returning empty string will count as a failure


def extract_sql_query(response):
    """Extract the SQL query from the model's response"""

    # Pattern 1: ```sql ... ```
    match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: ``` ... ```
    match = re.search(r"```\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 3: SELECT ... ; (with semicolon)
    match = re.search(r"(SELECT\s+.*?;)", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 4: SELECT ... (multiline, no semicolon)
    # Look for SELECT until we hit a newline followed by non-SQL text
    match = re.search(
        r"(SELECT\s+.+?)(?:\n\n|\n[A-Z][a-z]|\Z)", response, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # Pattern 5: Just the raw response if it starts with SELECT
    stripped = response.strip()
    if stripped.upper().startswith("SELECT"):
        return stripped

    # If all else fails
    print(
        f"--- WARNING: Could not extract SQL from response: ---\n{response}\n"
        + "-" * 50
    )
    return ""


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    """
    Save the logs of the experiment to files.
    You can change the format as needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(
            f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n"
        )
