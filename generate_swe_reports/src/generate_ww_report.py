"""
# TODO: docs
"""

from generate_maps import generate_maps
from generate_tables import generate_ww_tables
import argparse
from pathlib import Path
from jinja2 import Template
from datetime import datetime

def generate_ww_report(date: int) -> None:
    # TODO: docs

    PROJECT_ROOT = Path(__file__).parent.parent
    TEMPLATE_PATH = PROJECT_ROOT / "report_templates" / "TEMPLATE_WW_SWE_Report.tex"

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = Template(f.read())

    # TODO: config for changing output location
    maps_dir = PROJECT_ROOT / "output"  / f"{date}_WW_JPEGmaps"
    tables_dir = PROJECT_ROOT / "output"  / f"{date}_WW_TEXtables"

    context = {
        "instaar_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "INSTAAR_Logo.png").replace("\\", "/"),
        "bureau_reclamation_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "Bureau_Reclamation_Logo.png").replace("\\", "/"),
        "cu_boulder_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "CU_Boulder_Logo.png").replace("\\", "/"),
        "date": date,
        "date_string": datetime.strptime(str(date), "%Y%m%d").strftime("%B %d, %Y"),
        "fig1a_path" : str(maps_dir / f"{date}_WW_Fig1a.jpg").replace("\\", "/"),
        "fig1b_path": str(maps_dir / f"{date}_WW_Fig1b.jpg").replace("\\", "/"),
        "fig2a_path": str(maps_dir / f"{date}_WW_Fig2a.jpg").replace("\\", "/"),
        "fig2b_path": str(maps_dir / f"{date}_WW_Fig2b.jpg").replace("\\", "/"),
        "fig3_path": str(maps_dir / f"{date}_WW_Fig3.jpg").replace("\\", "/"),
        "fig4_path": str(maps_dir / f"{date}_WW_Fig4.jpg").replace("\\", "/"),
        "fig5_path": str(maps_dir / f"{date}_WW_Fig5.jpg").replace("\\", "/"),
        "fig6_path": str(maps_dir / f"{date}_WW_Fig6.jpg").replace("\\", "/"),
        "table1_path": str(tables_dir / f"{date}_WW_Table01.tex").replace("\\", "/"),
        "table2_path": str(tables_dir / f"{date}_WW_Table02.tex").replace("\\", "/"),
        "table3_path": str(tables_dir / f"{date}_WW_Table03.tex").replace("\\", "/"),
        "table4_path": str(tables_dir / f"{date}_WW_Table04.tex").replace("\\", "/"),
    }

    rendered_tex = template.render(**context)
    output_tex = PROJECT_ROOT / "output" / f"0WW_SWE_Report_{date}.tex"
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    print(f"Report written to {output_tex}")

def main():
    # Parse input arguments and flags, see top of file for argument usage examples
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=int, help="Date to process (YYYYMMDD)")
    parser.add_argument("--figs", default="all", type=str, help="Regex pattern(s) for figure names to generate")
    parser.add_argument("--tables", default="all", type=str, help="Regex pattern(s) for table ID's to generate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output messages")
    parser.add_argument("-u", "--prompt_user", action="store_true",
                        help="Prompt the user before overwriting or automatically selecting files")
    args = parser.parse_args()

    # Generate maps
    generate_maps("WW", args.date, args.figs, False, args.verbose, args.prompt_user)

    # Generate tables
    generate_ww_tables(args.date, args.tables, args.verbose)

    # Generate report
    generate_ww_report(args.date)

    #TODO: automatically compile LaTeX file (twice)

if __name__ == "__main__":
    main()
