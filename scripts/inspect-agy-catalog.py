"""Read tool declarations from a specific AGY conversation, without exporting prompts.

Observed AGY 1.1.28/1.2.0 gen_metadata wire layout: field 1 is the request;
request field 8 repeats tool declarations (name=1, description=2, JSON schema=3).
Unlike stream-json init.tools, this is the resolved per-generation model catalog.
This is a diagnostic for these versions, NOT a stable/public AGY API. Fail on
missing declarations or schema drift; never equate missing evidence with absence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

EXPECTED = {"view_file", "list_dir", "find_by_name", "grep_search", "manage_task"}
MAX_BLOB = 8 * 1024 * 1024


def varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("Truncated varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ValueError("Varint overflow")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("Invalid varint")


def fields(data):
    if len(data) > MAX_BLOB:
        raise ValueError("Metadata exceeds inspection budget")
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if not field:
            raise ValueError("Invalid field")
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = varint(data, offset)
            else:
                size = 8 if wire == 1 else 4
            if offset + size > len(data):
                raise ValueError("Truncated field")
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError("Unsupported wire type")
        yield field, wire, value


def messages(data, number):
    return [value for field, wire, value in fields(data) if field == number and wire == 2]


def catalog(blob):
    requests = messages(blob, 1)
    if not requests:
        return None  # AGY also stores non-generation metadata rows.
    if len(requests) != 1:
        raise ValueError("Ambiguous request")
    declarations = messages(requests[0], 8)
    if not declarations and not messages(requests[0], 1) and not messages(requests[0], 2):
        return None  # Older rows may retain usage only, with request content stripped.
    if not declarations:
        raise ValueError("Request has no tool declaration evidence")
    result = []
    for declaration in declarations:
        names = messages(declaration, 1)
        descriptions = messages(declaration, 2)
        schemas = messages(declaration, 3)
        if len(names) != 1 or len(descriptions) != 1 or len(schemas) != 1:
            raise ValueError("Unexpected tool declaration schema")
        schema = json.loads(schemas[0])
        if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
            raise ValueError("Invalid tool parameter schema")
        result.append(names[0].decode("utf-8"))
    if len(result) != len(set(result)):
        raise ValueError("Duplicate tool declaration")
    return sorted(result)


def inspect(database):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        count = db.execute("SELECT COUNT(*) FROM gen_metadata").fetchone()[0]
        if count > 256:
            raise ValueError("Too many generations for a bounded smoke")
        rows = []
        unavailable = []
        for index, size in db.execute("SELECT idx, length(data) FROM gen_metadata ORDER BY idx"):
            if size > MAX_BLOB:
                raise ValueError("Oversized metadata")
            blob = db.execute("SELECT data FROM gen_metadata WHERE idx=?", (index,)).fetchone()[0]
            tools = catalog(blob)
            if tools is not None:
                rows.append({"generation": index, "tools": tools,
                             "metadataSha256": hashlib.sha256(blob).hexdigest()})
            else:
                unavailable.append(index)
        if not rows:
            raise ValueError("No model catalog evidence")
        subtrajectories = db.execute(
            "SELECT COUNT(*) FROM steps WHERE has_subtrajectory <> 0"
        ).fetchone()[0]
    return {"conversationId": database.stem, "generations": rows,
            "generationsWithoutRetainedCatalog": unavailable,
            "stepsWithSubtrajectory": subtrajectories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--require-peer", action="store_true")
    args = parser.parse_args()
    result = inspect(args.database)
    if args.require_peer:
        if any(set(row["tools"]) != EXPECTED for row in result["generations"]):
            raise ValueError("Resolved catalog differs from the verified PEER set")
        if result["stepsWithSubtrajectory"]:
            raise ValueError("Unexpected subtrajectory in PEER smoke")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
