#!/usr/bin/env python3
import argparse
import os
import sys
from Ml import Ml

if __name__ == "__main__":
    scriptName = os.path.basename(__file__)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser( \
        prog = scriptName,
        description = "This scripts prepares the log data for machine learning.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--logdir", metavar="'logdir'", required=True, type=str,
                        help="The directory containing the logs")
    parser.add_argument("-o", "--outputdir", metavar="'outputdir'", required=True, type=str,
                        help="Output directory")
    parser.add_argument("-r", "--reportpath", metavar="'reportpath'", required=False, type=str,
                        help="The name/path of the report")
    args = parser.parse_args()
    client = Ml(logDir=args.logdir,
                reportPath=args.reportpath)
    client = Ml(logDir=args.logdir, outputDir=args.outputdir, reportPath=args.reportpath)
    rc = client.prepareData()
    sys.exit(rc)
