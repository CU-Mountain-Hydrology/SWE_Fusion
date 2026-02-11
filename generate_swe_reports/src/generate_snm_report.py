"""
# TODO: docs
"""

from generate_maps import get_output_dir as get_maps_dir
from generate_maps import generate_maps
from generate_swe_reports.src.utils import confirm_process
from generate_tables import get_output_dir as get_tables_dir
from generate_tables import generate_tables
import argparse
from pathlib import Path
from jinja2 import Template
from datetime import datetime
import subprocess

def generate_snm_report(date: int, verbose=False, prompt_user=False) -> Path:
    # TODO: docs

    PROJECT_ROOT = Path(__file__).parent.parent
    TEMPLATE_PATH = PROJECT_ROOT / "report_templates" / "TEMPLATE_SNM_SWE_Report.tex"

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = Template(f.read())

    # Location where the maps and tables are saved
    maps_dir = Path(get_maps_dir(date, 'SNM'))
    tables_dir = Path(get_tables_dir(date, 'SNM'))

    context = {
        "cu_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "CU_Logo_Notext.jpg").replace("\\", "/"),
        "instaar_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "INSTAAR_75_Logo.png").replace("\\", "/"),
        "date": date,
        "date_string": datetime.strptime(str(date), "%Y%m%d").strftime("%B %#d, %Y"),
        "fig0_path" : str(maps_dir / f"{date}_SNM_Fig0.jpg").replace("\\", "/"),
        "fig1_path": str(maps_dir / f"{date}_SNM_Fig1.jpg").replace("\\", "/"),
        "fig2_path": str(maps_dir / f"{date}_SNM_Fig2.jpg").replace("\\", "/"),
        "fig3_path": str(maps_dir / f"{date}_SNM_Fig3.jpg").replace("\\", "/"),
        "fig4_path": str(maps_dir / f"{date}_SNM_Fig4.jpg").replace("\\", "/"),
        "fig5_path": str(maps_dir / f"{date}_SNM_Fig5.jpg").replace("\\", "/"),
        "fig6_path": str(maps_dir / f"{date}_SNM_Fig6.jpg").replace("\\", "/"),
        "fig7_path": str(maps_dir / f"{date}_SNM_Fig7.jpg").replace("\\", "/"),
        "table5_path": str(tables_dir / f"{date}_SNM_Table05.tex").replace("\\", "/"),
        "table10_path": str(tables_dir / f"{date}_SNM_Table10.tex").replace("\\", "/"),
    }

    output_tex = Path(get_maps_dir(date, 'SNM')).parent / "LaTeX" / f"{date}_RT_SWE_Report.tex"

    # Abort if user denies file overwriting
    if prompt_user and output_tex.exists() and confirm_process(f"{date}_RT_SWE_Report.tex already exists and will be overwritten!"):
        pass

    rendered_tex = template.render(**context)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    if verbose: print(f"Report written to {output_tex}")
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
    figs_is_none = any(p.strip().lower() in {"none", ""} for p in args.figs.split(","))
    if not figs_is_none:
        generate_maps("SNM", args.date, args.figs, False, args.verbose, args.prompt_user)
    elif args.verbose:
        print(f"No figures will be generated: --figs={args.figs}")

    # Generate tables
    tables_is_none = any(p.strip().lower() in {"none", ""} for p in args.tables.split(","))
    if not tables_is_none:
        generate_tables("SNM", args.date, args.tables, args.verbose, args.prompt_user)
    elif args.verbose:
        print(f"No tables will be generated: --tables={args.tables}")

    # Generate report
    output_path = generate_snm_report(args.date, args.verbose, args.prompt_user)

    # Automatically compile LaTeX file (twice)
    # TODO: docs on how to set up pdflatex and get subprocess working
    report_dir = Path(get_maps_dir(args.date, 'SNM')).parent
    for _ in range(2):
        print("Compiling LaTeX to PDF.")
        subprocess.run(
            ["pdflatex",
             "-file-line-error",
             "-synctex=1",
             "-output-format=pdf",
             "-interaction=nonstopmode",
             f"-output-directory={report_dir / 'LaTeX'}",
             output_path.name],
            cwd=report_dir,
            check=True
        )
    # Move output PDF from LaTeX/ to Report/
    src = report_dir / "LaTeX" / f"{args.date}_RT_SWE_Report.pdf"
    dest = report_dir / f"{args.date}_RT_SWE_Report.pdf"
    src.replace(dest)

if __name__ == "__main__":
    main()
