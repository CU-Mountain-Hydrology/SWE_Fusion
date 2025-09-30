# TODO: docs
import argparse
import glob
import os
from pathlib import Path
import pandas as pd
from jinja2 import Template, Environment, FileSystemLoader
from generate_maps import generate_maps
from datetime import datetime

table_data = {
    "01" : "Pacific Northwest",
    "02" : "North Continental",
    "03" : "South Continental",
    "04a": "Intermountain",
    "04b": "Intermountain",
}

def generate_ww_report(date: int):
    # TODO: docs

    PROJECT_ROOT = Path(__file__).parent.parent
    TEMPLATE_PATH = PROJECT_ROOT / "report_templates" / "TEMPLATE_WW_SWE_Report.tex"

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = Template(f.read())

    maps_dir = PROJECT_ROOT / "output"  / f"{date}_WW_JPEGmaps"
    tables_dir = PROJECT_ROOT / "output"  / f"{date}_WW_TEXtables"

    context = {
        "instaar_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "INSTAAR_Logo.png").replace("\\", "/"),
        "bureau_reclamation_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "Bureau_Reclamation_Logo.png").replace("\\", "/"),
        "cu_boulder_logo_path": str(PROJECT_ROOT / "report_templates" / "images" / "CU_Boulder_Logo.png").replace("\\", "/"),
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
        "table4a_path": str(tables_dir / f"{date}_WW_Table04a.tex").replace("\\", "/"),
        "table4b_path": str(tables_dir / f"{date}_WW_Table04b.tex").replace("\\", "/"),
    }

    rendered_tex = template.render(**context)
    output_tex = PROJECT_ROOT / "output" / "weekly_report.tex"
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(rendered_tex)

    print(f"Report written to {output_tex}")

def main():
    # Parse input arguments and flags, see top of file for argument usage examples
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=int, help="Date to process (YYYYMMDD)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output messages")
    parser.add_argument("-u", "--prompt_user", action="store_true",
                        help="Prompt the user before overwriting or automatically selecting files")
    args = parser.parse_args()

    # Generate maps
    # TODO: only if JPEGmaps folder doesnt exist?
    # generate_maps("WW", args.date, "all", False, args.verbose, args.prompt_user)

    # Generate tables from CSV
    date_str = str(args.date)
    report_dir = fr"W:\documents\{date_str[:4]}_RT_Reports\{date_str}_RT_Report"
    use_this_dir = glob.glob(os.path.join(report_dir, "*UseThis"))[0]
    table_dir = os.path.join(use_this_dir, "Tables", "forUpload")
    output_tables_dir = Path(__file__).parent.parent / "output" / f"{args.date}_WW_TEXtables"
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    for table_id in list(table_data.keys()):
        matches = glob.glob(os.path.join(table_dir, f"*{table_id}.csv"))
        # TODO: error handling
        table = matches[0]

        df = pd.read_csv(table)
        headers = {
            "% of Average": ["3/15", "3/31"],
            "SWE (in)": ["3/15", "3/31"],
            "": ["SCA"],
            " ": ["Vol. (AF)"],
            "   ": ["Area (mi$^2$)"],
            "Pillows": ["3/15", "3/31"],
            "Surveys": ["3/31"],
        }
        templates_dir = Path(__file__).parent.parent / "report_templates"
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("TEMPLATE_SWE_Table.tex")

        latex_table = template.render(df=df, title=table_data[table_id], headers=headers)

        output_table = output_tables_dir / f"{args.date}_WW_Table{table_id}.tex"
        with open(output_table, "w") as f:
            f.write(latex_table)

    # Generate report
    generate_ww_report(args.date)

if __name__ == "__main__":
    main()
