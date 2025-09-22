#!/usr/bin/env python3
"""
Script to create visualizations for existing ENKF precomputed data.

Usage:
    python scripts/create_visualizations.py --config_dir logs/L96_1
    python scripts/create_visualizations.py --config_dir logs/L96_1 --force
"""

import os
import sys
import argparse

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from utils.data_generation import create_visualizations_for_existing_data


def main():
    parser = argparse.ArgumentParser(description='Create visualizations for ENKF precomputed data')
    parser.add_argument('--config_dir', required=True,
                       help='Configuration directory (e.g., logs/L96_1)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of visualizations even if they exist')

    args = parser.parse_args()

    # Convert to absolute path
    config_dir = os.path.abspath(args.config_dir)

    if not os.path.exists(config_dir):
        print(f"Error: Configuration directory {config_dir} does not exist")
        return 1

    print(f"Creating visualizations for configuration: {config_dir}")
    print(f"Force regeneration: {args.force}")

    try:
        create_visualizations_for_existing_data(config_dir, force_regenerate=args.force)
        print("Visualization creation completed successfully!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())