
import argparse
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_sun():
    from physics.sun_3d import generar_sol_metriplectico_3d
    print("Running Sun Metriplectic 3D Simulation...")
    generar_sol_metriplectico_3d()

def run_perihelium():
    from physics.perihelium import comparar_modelos
    print("Running Perihelion Calculation...")
    # Default to 5 years for quick run
    comparar_modelos(t_final=5*365.25*24*3600)

def run_validation():
    from docs.validation import run_validation_tests
    print("Running Validation Tests...")
    run_validation_tests()

def run_argue():
    from docs.argue import main as argue_main
    print("Running Argumentation Defense...")
    argue_main()

def run_analysis():
    from docs.analysis import main as analysis_main
    print("Running Analysis Report...")
    analysis_main()

def main():
    parser = argparse.ArgumentParser(description="Gravity Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommands
    subparsers.add_parser("sun", help="Run Sun Metriplectic visualization")
    subparsers.add_parser("perihelium", help="Run Perihelion precession calculation")
    subparsers.add_parser("validation", help="Run validation tests")
    subparsers.add_parser("argue", help="Run defense argumentation")
    subparsers.add_parser("analysis", help="Run comparative analysis")

    args = parser.parse_args()

    if args.command == "sun":
        run_sun()
    elif args.command == "perihelium":
        run_perihelium()
    elif args.command == "validation":
        run_validation()
    elif args.command == "argue":
        run_argue()
    elif args.command == "analysis":
        run_analysis()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
