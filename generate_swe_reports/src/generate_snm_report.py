"""
# TODO: docs
"""

from generate_maps import generate_maps
from generate_tables import generate_tables
import argparse
from pathlib import Path
from jinja2 import Template
from datetime import datetime
import subprocess

def generate_snm_report(date: int) -> Path:
    # TODO: docs

    PROJECT_ROOT = Path(__file__).parent.parent
    TEMPLATE_PATH = PROJECT_ROOT / "report_templates" / "TEMPLATE_SNM_SWE_Report_First.tex"

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = Template(f.read())

    # TODO: config for changing output location
    maps_dir = PROJECT_ROOT / "output"  / f"{date}_SNM_JPEGmaps"
    tables_dir = PROJECT_ROOT / "output"  / f"{date}_SNM_TEXtables"

    context = {
        "cu_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "CU_Logo_Notext.jpg").replace("\\", "/"),
        "instaar_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "INSTAAR_75_Logo.png").replace("\\", "/"),
        "date": date,
        "date_string": datetime.strptime(str(date), "%Y%m%d").strftime("%B %#d, %Y"),
        "fig0_path" : str(maps_dir / f"{date}_SNM_Fig0.jpg").replace("\\", "/"),
        "fig1_path": str(maps_dir / f"{date}_SNM_Fig1.jpg").replace("\\", "/"),
        # "fig2_path": str(maps_dir / f"{date}_SNM_Fig2.jpg").replace("\\", "/"),
        # "fig3_path": str(maps_dir / f"{date}_SNM_Fig3.jpg").replace("\\", "/"),
        "fig4_path": str(maps_dir / f"{date}_SNM_Fig4.jpg").replace("\\", "/"),
        "fig5_path": str(maps_dir / f"{date}_SNM_Fig5.jpg").replace("\\", "/"),
        "fig6_path": str(maps_dir / f"{date}_SNM_Fig6.jpg").replace("\\", "/"),
        "fig7_path": str(maps_dir / f"{date}_SNM_Fig7.jpg").replace("\\", "/"),
        "table5_path": str(tables_dir / f"{date}_SNM_Table5.tex").replace("\\", "/"),
        "table10_path": str(tables_dir / f"{date}_SNM_Table10.tex").replace("\\", "/"),
    }

    rendered_tex = template.render(**context)
    # TODO: config for output location
    output_tex = PROJECT_ROOT / "output" / f"{date}_RT_SWE_Report.tex"
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    print(f"Report written to {output_tex}")
    return output_tex

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
    # generate_maps("WW", args.date, args.figs, False, args.verbose, args.prompt_user)

    # Generate tables
    generate_tables("SNM", args.date, args.tables, args.verbose)

    # Generate report
    output_path = generate_snm_report(args.date)

    # Automatically compile LaTeX file (twice)
    # TODO: docs on how to set up pdflatex and get subprocess working
    for _ in range(2):
        print("Compiling LaTeX to PDF.")
        subprocess.run(
            ["pdflatex",
            "-file-line-error",
            "-synctex=1",
            "-output-format=pdf",
            "-interaction=nonstopmode",
            output_path.name],
            cwd = Path(__file__).parent.parent / "output", # TODO: make output path a config
            check=True
        )

if __name__ == "__main__":
    main()
