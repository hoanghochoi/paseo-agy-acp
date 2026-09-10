"""Negative tests for the catalog evidence reader (no AGY or user data)."""

import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "catalog_reader", Path(__file__).with_name("inspect-agy-catalog.py")
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def field(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def declaration(name):
    schema = json.dumps({"type": "object", "properties": {}}).encode()
    return field(1, name.encode()) + field(2, b"description") + field(3, schema)


class EvidenceTests(unittest.TestCase):
    def test_reads_explicit_declarations(self):
        blob = field(1, field(8, declaration("view_file")))
        self.assertEqual(reader.catalog(blob), ["view_file"])

    def test_does_not_hide_forbidden_tool(self):
        blob = field(1, field(8, declaration("invoke_subagent")))
        self.assertEqual(reader.catalog(blob), ["invoke_subagent"])

    def test_stripped_metadata_is_not_empty_catalog(self):
        self.assertIsNone(reader.catalog(field(1, field(19, b"model"))))

    def test_full_request_without_tools_fails(self):
        with self.assertRaises(ValueError):
            reader.catalog(field(1, field(1, b"system prompt")))

    def test_truncation_fails(self):
        with self.assertRaises(ValueError):
            reader.catalog(field(1, field(8, declaration("view_file")))[:-1])

    def test_duplicate_tools_fail(self):
        tool = field(8, declaration("view_file"))
        with self.assertRaises(ValueError):
            reader.catalog(field(1, tool + tool))

    def test_invalid_json_schema_fails(self):
        tool = field(1, b"view_file") + field(2, b"description") + field(3, b"{}")
        with self.assertRaises(ValueError):
            reader.catalog(field(1, field(8, tool)))


if __name__ == "__main__":
    unittest.main()
