#!/usr/bin/env python3
"""
Generate data for all L96 configurations in parallel using multithreading.

Usage:
    python generate_data_parallel.py --dir=logs [--force-regenerate]
"""

import os
import sys
import argparse
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_data_generation(subdir, force_regenerate=False):
    """Run data generation for a single subdirectory"""
    script_path = os.path.join(os.path.dirname(__file__), "generate_data.py")

    cmd = [sys.executable, script_path, f"--dir={subdir}"]
    if force_regenerate:
        cmd.append("--force-regenerate")

    try:
        print(f"[{threading.current_thread().name}] Starting data generation for {subdir}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[{threading.current_thread().name}] ✓ Completed {subdir}")
        return subdir, 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"[{threading.current_thread().name}] ✗ Failed {subdir}: {e}")
        return subdir, e.returncode, e.stdout, e.stderr
    except Exception as e:
        print(f"[{threading.current_thread().name}] ✗ Error in {subdir}: {e}")
        return subdir, 1, "", str(e)

def main():
    parser = argparse.ArgumentParser(description='Generate data for all L96 configurations in parallel')
    parser.add_argument('--dir', required=True, help='Base directory containing L96_* subdirectories')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration of existing files')

    args = parser.parse_args()
    base_dir = args.dir

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)

    # Find all L96_* subdirectories
    subdirs = []
    for item in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, item)
        if os.path.isdir(subdir_path) and item.startswith("L96_"):
            config_path = os.path.join(subdir_path, "config.py")
            if os.path.exists(config_path):
                subdirs.append(subdir_path)
            else:
                print(f"Warning: {subdir_path} missing config.py, skipping")

    if not subdirs:
        print(f"No L96_* directories with config.py found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(subdirs)} directories to process:")
    for subdir in subdirs:
        print(f"  {subdir}")

    # Run data generation in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(subdirs), thread_name_prefix="DataGen") as executor:
        # Submit all tasks
        future_to_subdir = {
            executor.submit(run_data_generation, subdir, args.force_regenerate): subdir
            for subdir in subdirs
        }

        # Process completed tasks
        for future in as_completed(future_to_subdir):
            subdir, exit_code, stdout, stderr = future.result()
            results.append((subdir, exit_code, stdout, stderr))

    # Report results
    print("\n" + "="*50)
    print("Data generation summary:")

    success_count = 0
    failure_count = 0

    for subdir, exit_code, stdout, stderr in results:
        if exit_code == 0:
            status = "✓ SUCCESS"
            success_count += 1
        else:
            status = "✗ FAILED"
            failure_count += 1
            if stderr:
                print(f"  {subdir}: {status}")
                print(f"    Error: {stderr}")
            else:
                print(f"  {subdir}: {status}")

    print(f"\nTotal: {success_count} succeeded, {failure_count} failed")

    if failure_count > 0:
        print("\nSome data generation tasks failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll data generation completed successfully!")

if __name__ == "__main__":
    main()