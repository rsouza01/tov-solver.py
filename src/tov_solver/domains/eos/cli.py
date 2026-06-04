import argparse


def handle_polytrope(args):
	# Implementation logic to generate a polytropic EOS
	print(f"Generating polytropic EOS with Gamma={args.gamma} and K={args.kappa}")


# e.g., models.generate_polytrope(args.gamma, args.kappa)

def handle_tabulated(args):
	# Implementation logic for loading tabulated crust/core data
	print(f"Loading tabulated EOS data from {args.file}")


def register_commands(subparsers: argparse._SubParsersAction):
	# 1. Register the 'eos' domain
	eos_parser = subparsers.add_parser("eos", help="Equation of State operations")

	# 2. Create subcommands under 'eos'
	eos_subparsers = eos_parser.add_subparsers(
		title="commands", dest="command", required=True
	)

	# Command: tov-solver.py eos polytrope --gamma 2.0 --kappa 100.0
	poly_parser = eos_subparsers.add_parser("polytrope", help="Generate a polytropic EOS")
	poly_parser.add_argument("--gamma", type=float, required=True, help="Polytropic index")
	poly_parser.add_argument("--kappa", type=float, required=True, help="Proportionality constant")
	poly_parser.set_defaults(func=handle_polytrope)

	# Command: tov-solver.py eos tabulated --file data.csv
	tab_parser = eos_subparsers.add_parser("tabulated", help="Load tabulated EOS data")
	tab_parser.add_argument("--file", type=str, required=True, help="Path to EOS data file")
	tab_parser.set_defaults(func=handle_tabulated)
