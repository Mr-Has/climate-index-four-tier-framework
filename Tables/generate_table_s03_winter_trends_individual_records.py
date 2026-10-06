"""Reproduce table s03. Run with --help for output options.

Calculation and rendering code: ../Code/reproduce.py
Source workbook: ../Data/Climate_Indices_Data.xlsx
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Code'))
from reproduce import main

if __name__ == '__main__':
    main('table_s03')
