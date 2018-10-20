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
        description = "This scripts builds supervised machine learning models",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--cfgpath", metavar="'cfgpath'", required=False, type=str,
                        default="{0}/ml.json".format(dirPath), help="The JSON configuration file")
    parser.add_argument("-D", "--isdirectory", action="store_true",
                        help="Input all data CSV files in given directory specified by using -d <dir>")
    parser.add_argument("-d", "--inputdatapath", metavar="'inputdatapath'", required=True, type=str,
                        help="Input given data CSV file or if -D is used all CSV files under -d <dir>")
    parser.add_argument("-k", "--kfolds", metavar="'kfolds'", required=False, type=int,
                        default=2, help="The number of kfolds for train/test")
    parser.add_argument("-o", "--outputdir", metavar="'outputdir'", required=True, type=str,
                        help="Output directory")
    args = parser.parse_args()

    client = Ml.ModelGeneration(outputDir=args.outputdir, cfgPath=args.cfgpath, kFolds=args.kfolds)
    sys.exit(client.run(inputDataPath=args.inputdatapath, isDirectoryFlag=args.isdirectory))
