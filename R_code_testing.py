import os
import sys
import rpy2
os.environ['R_HOME'] = 'C:/PROGRA~1/R/R-41~1.3'

from pathlib import Path


# Step 1: Install rpy2 if not already installed
# Step 1: Install rpy2 if not already installed
def check_and_install_rpy2():
    """Check if rpy2 is installed, provide installation instructions if not"""
    try:
        # Try to get version (works differently in different rpy2 versions)
        try:
            import rpy2
            version = rpy2.__version__
        except AttributeError:
            try:
                from rpy2 import __version__ as version
            except ImportError:
                version = "unknown"

        print(f"✓ rpy2 is installed (version {version})")
        return True  # <-- ADD THIS LINE
    except ImportError:
        print("✗ rpy2 is not installed")
        print("\nTo install rpy2, run:")
        print("  pip install rpy2")
        print("\nFor Windows, you may need:")
        print("  pip install rpy2-windows")
        return False


# Step 2: Set up R environment
def setup_r_environment():
    """
    Configure R environment for rpy2
    This ensures rpy2 can find your R installation
    """

    # Check if R_HOME is already set
    if 'R_HOME' in os.environ:
        print(f"✓ R_HOME is set: {os.environ['R_HOME']}")
        return True

    # Common R installation paths by platform
    if sys.platform == 'win32':
        # Windows - check common installation paths
        possible_paths = [
            'C:/Program Files/R/R-4.3.0',
            'C:/Program Files/R/R-4.2.0',
            'C:/Program Files/R/R-4.1.0',
        ]
    elif sys.platform == 'darwin':
        # macOS
        possible_paths = [
            '/Library/Frameworks/R.framework/Resources',
            '/usr/local/lib/R',
        ]
    else:
        # Linux
        possible_paths = [
            '/usr/lib/R',
            '/usr/local/lib/R',
        ]

    # Try to find R installation
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['R_HOME'] = path
            print(f"✓ R_HOME set to: {path}")
            return True

    print("✗ Could not find R installation automatically")
    print("\nPlease set R_HOME manually:")
    print("  import os")
    print("  os.environ['R_HOME'] = '/path/to/R'")
    return False


# Step 3: Import rpy2 components
def import_rpy2_components():
    """Import necessary rpy2 components"""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        # Activate automatic pandas conversion
        pandas2ri.activate()

        print("✓ rpy2 components imported successfully")
        return ro, pandas2ri, importr

    except Exception as e:
        print(f"✗ Error importing rpy2: {e}")
        return None, None, None


# Step 4: Check if required R packages are installed
def check_r_packages(ro, importr):
    """Check if required R packages are installed"""

    required_packages = [
        'stationsweRegression',
        'StationSWERegressionV2',
        'readr',  # For read_csv
    ]

    print("\nChecking R packages:")

    missing_packages = []
    for pkg in required_packages:
        try:
            importr(pkg)
            print(f"  ✓ {pkg} is installed")
        except Exception:
            print(f"  ✗ {pkg} is NOT installed")
            missing_packages.append(pkg)

    if missing_packages:
        print("\n⚠ Missing packages detected!")
        print("\nTo install them, open R or RStudio and run:")
        for pkg in missing_packages:
            print(f"  install.packages('{pkg}')")
        print("\nOr install from Python:")
        print("  utils = importr('utils')")
        print("  utils.install_packages('" + "', '".join(missing_packages) + "')")
        return False

    return True


# Step 5: Main function to run the R script
def run_r_script_with_rpy2(r_script_path, simdate='2025-12-11', path_root=None):
    """
    Execute R script using rpy2 with parameter customization

    Args:
        r_script_path: Path to the R script file
        simdate: Simulation date in 'YYYY-MM-DD' format
        path_root: Root path for data storage (optional override)

    Returns:
        bool: True if successful, False otherwise
    """

    print(f"\n{'=' * 70}")
    print(f"Running R script with rpy2")
    print(f"{'=' * 70}")

    # Import rpy2
    import rpy2.robjects as ro

    # Read the R script
    print(f"\n1. Reading R script: {r_script_path}")

    if not os.path.exists(r_script_path):
        print(f"✗ Error: R script not found at {r_script_path}")
        return False

    with open(r_script_path, 'r') as f:
        r_code = f.read()

    print(f"   ✓ Script loaded ({len(r_code)} characters)")

    # Modify parameters if requested
    print(f"\n2. Setting parameters:")
    print(f"   simdate: {simdate}")

    # Replace simdate in the R code
    r_code = r_code.replace(
        "simdate <- '2025-12-11'",
        f"simdate <- '{simdate}'"
    )

    # Optionally replace path_root
    if path_root:
        print(f"   path_root: {path_root}")
        r_code = r_code.replace(
            "PATH_root <- paste0('H:/WestUS_Data/Regress_SWE/')",
            f"PATH_root <- '{path_root}'"
        )

    # Execute the R code
    print(f"\n3. Executing R code...")
    print(f"{'=' * 70}")

    try:
        # Run the R code - this executes everything in the script
        ro.r(r_code)

        print(f"{'=' * 70}")
        print("✓ R script executed successfully!")
        return True

    except Exception as e:
        print(f"{'=' * 70}")
        print(f"✗ Error executing R script:")
        print(f"   {str(e)}")
        return False


# Main execution
def main():
    """Main function demonstrating all methods"""

    print("=" * 70)
    print("R Script Execution via rpy2 - Complete Implementation")
    print("=" * 70)

    # Step 1: Check rpy2 installation
    if not check_and_install_rpy2():
        print("\n⚠ Please install rpy2 first and re-run this script")
        return

    # Step 2: Setup R environment
    if not setup_r_environment():
        print("\n⚠ Please set R_HOME environment variable")
        return

    # Step 3: Import rpy2
    ro, pandas2ri, importr = import_rpy2_components()
    if ro is None:
        return

    # Step 4: Check R packages
    if not check_r_packages(ro, importr):
        print("\n⚠ Please install missing R packages")
        # Optionally continue anyway if you want to test

    # Step 5: Run the R script
    # Update this path to your actual R script location
    r_script_path = "/mnt/user-data/uploads/0_get_All_stationswe_data_OLAF.R"

    print(f"\n{'=' * 70}")
    print("Choose a method to run:")
    print("=" * 70)
    print("\n1. METHOD 1: Source the file directly (simplest)")
    print("   - No modifications to the script")
    print("   - Uses parameters as defined in the R file")

    print("\n2. METHOD 2: Run with parameter modifications (recommended)")
    print("   - Customize simdate and path_root")
    print("   - More flexible")

    print("\n3. METHOD 3: Run with output capture (advanced)")
    print("   - Access R variables from Python")
    print("   - Convert R dataframes to pandas")


    #METHOD 2: With modifications (uncomment to use)
    print("\n" + "="*70)
    print("Running METHOD 2...")
    print("="*70)
    success = run_r_script_with_rpy2(
        r_script_path,
        simdate='2025-12-12',
        path_root='H:/WestUS_Data/Regress_SWE/'
    )

if __name__ == "__main__":
    main()