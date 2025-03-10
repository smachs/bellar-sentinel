#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix all instances of incorrect get() calls in the codebase.
"""

import os
import re

def fix_get_calls(file_path):
    """Fix get() calls in a file to use the default parameter."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match get() calls with a third positional argument (not using default=)
    # Specifically looking for .get('key1', default= 'key2', default=default_value) pattern
    pattern = r'\.get\(([\'"][^\'",]+[\'"])[\s]*,[\s]*([\'"][^\'",]+[\'"])[\s]*,[\s]*([^,)]+)(?!\s*=)'
    
    # Replace with .get('key1', default= 'key2', default=default=default_value)
    fixed_content = re.sub(pattern, r'.get(\1, \2, default=\3', content)
    
    # Also find .get('key', default=default_value) pattern (for dict.get() usage)
    dict_pattern = r'\.get\(([\'"][^\'",]+[\'"])[\s]*,[\s]*([^,{)\'"]+)(?!\s*=|\s*{)'
    
    # Only replace if it's not already using default=
    def dict_replacer(match):
        key = match.group(1)
        value = match.group(2)
        if 'default=' not in value:
            return f'.get({key}, default={value}'
        return match.group(0)
    
    fixed_content = re.sub(dict_pattern, dict_replacer, fixed_content)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed {file_path}")

def find_python_files(directory):
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def main():
    """Main function."""
    # Find all Python files in the project
    python_files = find_python_files('.')
    
    # Fix get() calls in each file
    for file_path in python_files:
        fix_get_calls(file_path)
    
    print("All files processed successfully!")

if __name__ == '__main__':
    main() 