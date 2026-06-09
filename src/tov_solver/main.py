#!/usr/bin/env python3

import argparse
import sys

# Import the CLI registrars from each domain
from tov_solver.domains.eos import cli as eos_cli
from tov_solver.domains.tov import cli as tov_cli

def main():
    parser = argparse.ArgumentParser(
        prog="tov_solver",
        description="CLI suite for stellar evolution and compact object simulations."
    )

    # Create the top-level domain router (e.g., eos, tov)
    subparsers = parser.add_subparsers(
        title="domains",
        dest="domain",
        required=True,
        help="Target simulation domain"
    )

    # Delegate command registration to the specific domain packages
    eos_cli.register_commands(subparsers)
    tov_cli.register_commands(subparsers)

    args = parser.parse_args()

    # Automatic dispatch: execute the function bound to the specific subcommand
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
