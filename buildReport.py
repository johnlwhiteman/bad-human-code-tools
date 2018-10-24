#!/usr/bin/env python3
import argparse
import os
import sys
import Ml

if __name__ == "__main__":
    scriptName = os.path.basename(__file__)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser( \
        prog = scriptName,
        description = "This script builds a report from the machine learning models.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inputdatapath", metavar="'inputdatapath'", required=True, type=str,
                        help="Input given data CSV file")
    parser.add_argument("-o", "--outputdatapath", metavar="'outputdatapath'", required=True, type=str,
                        help="Output csv file")
    args = parser.parse_args()

    client = Ml.ReportBuilder(inputDataPath=args.inputdatapath, outputDataPath=args.outputdatapath)
    sys.exit(client.run())
