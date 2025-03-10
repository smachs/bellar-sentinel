#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix all incorrect ConfigLoader.get() calls in the codebase.
"""

import os
import re

def fix_file(file_path):
    """Fix ConfigLoader.get() calls in a file."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Check for config.get calls with 3 positional arguments
        # Example: self.config.get('sentiment', 'transformer_model', 'distilbert...')
        match = re.search(r'(self\.config|config)\.get\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"],\s*([^,)]+)\)', line)
        if match:
            prefix = match.group(1)
            first_arg = match.group(2)
            second_arg = match.group(3)
            third_arg = match.group(4)
            
            # Replace with correct usage
            fixed = f"{prefix}.get('{first_arg}', '{second_arg}', default={third_arg})"
            line = line.replace(match.group(0), fixed)
        
        fixed_lines.append(line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {file_path}")

def main():
    """Main function to fix files."""
    # Key files we know have issues
    files_to_fix = [
        os.path.join('app', 'core', 'sentiment_analyzer.py'),
        os.path.join('app', 'core', 'market_monitor.py'),
        os.path.join('app', 'core', 'defcon_system.py'),
        os.path.join('app', 'utils', 'logger.py'),
        'app.py'
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            fix_file(file_path)
    
    print("All files fixed successfully!")

if __name__ == '__main__':
    main() 