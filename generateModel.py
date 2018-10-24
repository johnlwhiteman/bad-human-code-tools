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
        description = "This script generates the supervised machine learning models",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inputdatapath", metavar="'inputdatapath'", required=True, type=str,
                        help="Input given data CSV file")
    parser.add_argument("-k", "--kfolds", metavar="'kfolds'", required=False, type=int,
                        default=2, help="The number of kfolds for train/test")
    parser.add_argument("-o", "--outputdatapath", metavar="'outputdatapath'", required=True, type=str,
                        help="Output csv file")
    parser.add_argument("-R", "--randomizedata", action="store_true",
                        help="Randomize the data")
    args = parser.parse_args()

    client = Ml.ModelGeneration(inputDataPath=args.inputdatapath, outputDataPath=args.outputdatapath,
                                kFolds=args.kfolds, randomizeDataFlag=args.randomizedata,
                                cycles=args.cycles)
    sys.exit(client.run())
