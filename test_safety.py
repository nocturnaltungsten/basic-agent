#!/usr/bin/env python3
"""Test script for safety features"""

import tempfile
import os
import subprocess


def test_safe_delete():
    """Test that delete operations move to trash instead of permanent deletion"""

    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".test") as f:
        test_file = f.name
        f.write("Test content for safety check")

    print(f"Created test file: {test_file}")

    # Test our safe delete function
    from main import safe_delete_files

    # Create a find pattern for our test file
    find_pattern = f"{test_file} -type f"
    result = safe_delete_files(find_pattern)

    print(f"Delete result: {result}")

    # Check if file is gone from original location
    if not os.path.exists(test_file):
        print("✅ File successfully moved from original location")
    else:
        print("❌ File still exists in original location")

    # Check if file is in trash (macOS)
    if os.path.exists(os.path.expanduser("~/.Trash")):
        trash_contents = subprocess.run("ls ~/.Trash/", shell=True, capture_output=True, text=True)
        if os.path.basename(test_file) in trash_contents.stdout:
            print("✅ File found in Trash")
        else:
            print("❌ File not found in Trash")


def test_destructive_command_detection():
    """Test that destructive commands are properly detected and classified"""
    from main import is_destructive_command, classify_command_risk

    destructive_commands = [
        ("rm -rf /tmp/test", "medium"),
        ("mv file1 file2", "medium"),
        ("dd if=/dev/zero of=/tmp/test", "high"),
        ("chmod 777 /etc/passwd", "high"),
        ("rm -rf /", "high"),
        ("rm file.txt", "low"),
    ]

    safe_commands = ["ls -la", "cat /etc/hosts", "echo hello", "ps aux"]

    print("\nTesting destructive command detection and risk classification:")
    for cmd, expected_risk in destructive_commands:
        is_destructive = is_destructive_command(cmd)
        actual_risk = classify_command_risk(cmd) if is_destructive else "safe"

        if is_destructive and actual_risk == expected_risk:
            print(f"✅ Correctly identified: {cmd} -> {actual_risk} risk")
        else:
            print(f"❌ Misclassified: {cmd} -> expected {expected_risk}, got {actual_risk}")

    print("\nTesting safe command detection:")
    for cmd in safe_commands:
        if not is_destructive_command(cmd):
            print(f"✅ Correctly identified as safe: {cmd}")
        else:
            print(f"❌ Incorrectly identified as destructive: {cmd}")


def test_confirmation_system_mock():
    """Test confirmation system with mocked input (non-interactive)"""
    from main import classify_command_risk

    test_commands = [
        ("rm -rf /important/data", "high"),
        ("rm temp.txt", "low"),
        ("mv oldfile newfile", "medium"),
    ]

    print("\nTesting risk classification system:")
    for cmd, expected_risk in test_commands:
        actual_risk = classify_command_risk(cmd)
        if actual_risk == expected_risk:
            print(f"✅ Risk classification correct: '{cmd}' -> {actual_risk}")
        else:
            print(
                f"❌ Risk classification wrong: '{cmd}' -> expected {expected_risk}, got {actual_risk}"
            )

    print(
        "\n💡 Note: Interactive confirmation testing requires manual testing in actual agent use."
    )


if __name__ == "__main__":
    print("🔒 Testing Safety Features")
    print("=" * 50)

    test_destructive_command_detection()
    print()
    test_confirmation_system_mock()
    print()
    test_safe_delete()

    print("\n✅ Safety tests completed!")
    print("\n💡 To test interactive confirmations:")
    print("   Run the agent and try: 'Run rm -rf /tmp/testdir'")
    print("   You should see confirmation prompts with risk levels.")
