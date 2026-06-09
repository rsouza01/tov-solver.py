import argparse


def handle_integrate(args):
	# Implementation for the core hydrostatic equilibrium integration
	print(f"Integrating TOV equations with central pressure Pc={args.pc} using EOS '{args.eos}'")


# e.g., integrator.run_tov(args.pc, args.eos)

def register_commands(subparsers: argparse._SubParsersAction):
	# 1. Register the 'tov' domain
	tov_parser = subparsers.add_parser("tov", help="Tolman-Oppenheimer-Volkoff solver")

	# 2. Create subcommands under 'tov'
	tov_subparsers = tov_parser.add_subparsers(
		title="commands", dest="command", required=True
	)

	# Command: tov-solver.py tov integrate --pc 1e35 --eos SLy
	int_parser = tov_subparsers.add_parser("integrate", help="Integrate hydrostatic equilibrium equations")
	int_parser.add_argument("--pc", type=float, required=True, help="Central pressure")
	int_parser.add_argument("--eos", type=str, required=True, help="Identifier of the EOS to use")
	int_parser.set_defaults(func=handle_integrate)
