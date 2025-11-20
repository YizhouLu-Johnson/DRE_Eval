#!/usr/bin/env python3
"""
Helper script to migrate TensorFlow 1.x code to TensorFlow 2.x with compat.v1 API.

This script will:
1. Find all Python files that import tensorflow
2. Replace `import tensorflow as tf` with `import tensorflow.compat.v1 as tf`
3. Add `tf.disable_v2_behavior()` after the import

Usage:
    python migrate_to_tf2.py [--dry-run] [--file FILE]
    
Options:
    --dry-run    Show what would be changed without making changes
    --file FILE  Process only the specified file (otherwise processes all .py files)
"""

import os
import re
import sys
import argparse


def process_file(filepath, dry_run=False):
    """Process a single Python file to add TF2 compatibility."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    original_content = content
    modified = False
    
    # Check if file imports tensorflow
    if 'import tensorflow' not in content:
        return False
    
    # Skip if already using compat.v1
    if 'tensorflow.compat.v1' in content or 'tf.compat.v1' in content:
        print(f"✓ {filepath} already uses compat.v1")
        return False
    
    # Pattern 1: import tensorflow as tf
    pattern1 = r'^(\s*)(import tensorflow as tf)(\s*)$'
    replacement1 = r'\1import tensorflow.compat.v1 as tf\3\n\1tf.disable_v2_behavior()\3'
    content_new = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
    
    if content_new != content:
        modified = True
        content = content_new
    
    # Pattern 2: from tensorflow import ...
    # We need to add the disable_v2_behavior after the imports
    pattern2 = r'^(\s*)(from tensorflow import .+)$'
    matches = list(re.finditer(pattern2, content, flags=re.MULTILINE))
    
    if matches and 'tf.disable_v2_behavior()' not in content:
        # Find the last tensorflow import
        last_match = matches[-1]
        end_pos = last_match.end()
        
        # Find the indentation
        indent = last_match.group(1)
        
        # Insert the compatibility code after the last tensorflow import
        content = (content[:end_pos] + 
                  f'\n{indent}import tensorflow.compat.v1 as tf\n{indent}tf.disable_v2_behavior()' + 
                  content[end_pos:])
        modified = True
    
    if modified:
        if dry_run:
            print(f"Would modify: {filepath}")
            return True
        else:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ Modified: {filepath}")
                return True
            except Exception as e:
                print(f"Error writing {filepath}: {e}")
                return False
    
    return False


def find_python_files(root_dir, exclude_dirs=None):
    """Find all Python files in the directory."""
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.git', 'venv', 'env', '.venv', 'notebooks'}
    
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories from the search
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files


def main():
    parser = argparse.ArgumentParser(
        description='Migrate TensorFlow 1.x code to use TF2 compat.v1 API'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be changed without making changes'
    )
    parser.add_argument(
        '--file', 
        type=str, 
        help='Process only the specified file'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Process single file
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            sys.exit(1)
        
        modified = process_file(args.file, dry_run=args.dry_run)
        if modified:
            if args.dry_run:
                print(f"\nDry run complete. Run without --dry-run to apply changes.")
            else:
                print(f"\nSuccessfully modified {args.file}")
        else:
            print(f"\nNo changes needed for {args.file}")
    else:
        # Process all Python files in current directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        python_files = find_python_files(script_dir)
        
        print(f"Found {len(python_files)} Python files")
        if args.dry_run:
            print("Running in dry-run mode...\n")
        
        modified_count = 0
        for filepath in python_files:
            if process_file(filepath, dry_run=args.dry_run):
                modified_count += 1
        
        print(f"\n{'Would modify' if args.dry_run else 'Modified'} {modified_count} file(s)")
        if args.dry_run and modified_count > 0:
            print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()

