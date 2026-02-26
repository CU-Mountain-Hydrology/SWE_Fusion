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

REPORT_CONFIGS = {
    "WW": {
        "template": "TEMPLATE_WW_SWE_Report.tex",
        "output_name": lambda date: f"0WW_SWE_Report_{date}",
        "context": lambda date, maps_dir, tables_dir, project_root: {
            "instaar_logo_path": str(project_root / "report_templates" / "images" / "INSTAAR_75_Logo.png").replace("\\", "/"),
            "bureau_reclamation_logo_path": str(project_root / "report_templates" / "images" / "Bureau_Reclamation_Logo.png").replace("\\", "/"),
            "cu_boulder_logo_path": str(project_root / "report_templates" / "images" / "CU_Boulder_Logo.png").replace("\\", "/"),
            "date": date,
            "date_string": datetime.strptime(str(date), "%Y%m%d").strftime("%B %#d, %Y"),
            "year": int(str(date)[0:4]),
            "fig1a_path": str(maps_dir["WW"] / f"{date}_WW_Fig1a.jpg").replace("\\", "/"),
            "fig1b_path": str(maps_dir["WW"] / f"{date}_WW_Fig1b.jpg").replace("\\", "/"),
            "fig2a_path": str(maps_dir["WW"] / f"{date}_WW_Fig2a.jpg").replace("\\", "/"),
            "fig2b_path": str(maps_dir["WW"] / f"{date}_WW_Fig2b.jpg").replace("\\", "/"),
            "fig3_path":  str(maps_dir["WW"] / f"{date}_WW_Fig3.jpg").replace("\\", "/"),
            "fig4_path":  str(maps_dir["WW"] / f"{date}_WW_Fig4.jpg").replace("\\", "/"),
            "fig5_path":  str(maps_dir["WW"] / f"{date}_WW_Fig5.jpg").replace("\\", "/"),
            "fig6_path":  str(maps_dir["WW"] / f"{date}_WW_Fig6.jpg").replace("\\", "/"),
            "fig7_path":  str(maps_dir["SNM"] / f"{date}_SNM_Fig1.jpg").replace("\\", "/"),
            "table1_path": str(tables_dir["WW"] / f"{date}_WW_Table01.tex").replace("\\", "/"),
            "table2_path": str(tables_dir["WW"] / f"{date}_WW_Table02.tex").replace("\\", "/"),
            "table3_path": str(tables_dir["WW"] / f"{date}_WW_Table03.tex").replace("\\", "/"),
            "table4_path": str(tables_dir["WW"] / f"{date}_WW_Table04.tex").replace("\\", "/"),
            "table5_path": str(tables_dir["SNM"] / f"{date}_SNM_Table05.tex").replace("\\", "/"),
        },
    },
    "SNM": {
        "template": "TEMPLATE_SNM_SWE_Report.tex",
        "output_name": lambda date: f"{date}_RT_SWE_Report",
        "context": lambda date, maps_dir, tables_dir, project_root: {
            "cu_logo_path": str(project_root / "report_templates" / "images" / "CU_Logo_Notext.jpg").replace("\\", "/"),
            "instaar_logo_path": str(project_root / "report_templates" / "images" / "INSTAAR_75_Logo.png").replace("\\", "/"),
            "date": date,
            "date_string": datetime.strptime(str(date), "%Y%m%d").strftime("%B %#d, %Y"),
            "fig0_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig0.jpg").replace("\\", "/"),
            "fig1_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig1.jpg").replace("\\", "/"),
            "fig2_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig2.jpg").replace("\\", "/"),
            "fig3_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig3.jpg").replace("\\", "/"),
            "fig4_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig4.jpg").replace("\\", "/"),
            "fig5_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig5.jpg").replace("\\", "/"),
            "fig6_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig6.jpg").replace("\\", "/"),
            "fig7_path": str(maps_dir["SNM"] / f"{date}_SNM_Fig7.jpg").replace("\\", "/"),
            "table5_path":  str(tables_dir["SNM"] / f"{date}_SNM_Table05.tex").replace("\\", "/"),
            "table10_path": str(tables_dir["SNM"] / f"{date}_SNM_Table10.tex").replace("\\", "/"),
        },
    },
}

def generate_report(report_type: str, date: int, verbose=False, prompt_user=False) -> Path:
    config = REPORT_CONFIGS[report_type]
    PROJECT_ROOT = Path(__file__).parent.parent

    maps_dir   = {rt: Path(get_maps_dir(date, rt))   for rt in ("WW", "SNM")}
    tables_dir = {rt: Path(get_tables_dir(date, rt)) for rt in ("WW", "SNM")}

    template_path = PROJECT_ROOT / "report_templates" / config["template"]
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())

    output_name = config["output_name"](date)
    output_tex = maps_dir[report_type].parent / "LaTeX" / f"{output_name}.tex"

    if prompt_user and output_tex.exists():
        confirm_process(f"{output_tex.name} already exists and will be overwritten!")

    context = config["context"](date, maps_dir, tables_dir, PROJECT_ROOT)
    rendered_tex = template.render(**context)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    if verbose:
        print(f"Report written to {output_tex}")
    return output_tex

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_type", type=str, choices=REPORT_CONFIGS.keys(), help="Report type (WW, SNM)")
    parser.add_argument("date", type=int, help="Date to process (YYYYMMDD)")
    parser.add_argument("--figs", default="all", type=str, help="Regex pattern(s) for figure names to generate")
    parser.add_argument("--tables", default="all", type=str, help="Regex pattern(s) for table ID's to generate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output messages")
    parser.add_argument("-u", "--prompt_user", action="store_true",
                        help="Prompt the user before overwriting or automatically selecting files")
    args = parser.parse_args()

    figs_is_none = any(p.strip().lower() in {"none", ""} for p in args.figs.split(","))
    if not figs_is_none:
        generate_maps(args.report_type, args.date, args.figs, False, args.verbose, args.prompt_user)
    elif args.verbose:
        print(f"No figures will be generated: --figs={args.figs}")

    tables_is_none = any(p.strip().lower() in {"none", ""} for p in args.tables.split(","))
    if not tables_is_none:
        generate_tables(args.report_type, args.date, args.tables, args.verbose, args.prompt_user)
    elif args.verbose:
        print(f"No tables will be generated: --tables={args.tables}")

    output_path = generate_report(args.report_type, args.date, args.verbose, args.prompt_user)

    report_dir = Path(get_maps_dir(args.date, args.report_type)).parent
    output_name = REPORT_CONFIGS[args.report_type]["output_name"](args.date)
    for _ in range(3):
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

    src  = report_dir / "LaTeX" / f"{output_name}.pdf"
    dest = report_dir / f"{output_name}.pdf"
    src.replace(dest)

if __name__ == "__main__":
    main()