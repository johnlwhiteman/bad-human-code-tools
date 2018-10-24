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
        description = "This scripts prepares the log data for machine learning.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--cfgpath", metavar="'cfgpath'", required=False, type=str,
                        default="{0}/ml.json".format(dirPath), help="The JSON configuration file")
    parser.add_argument("-l", "--logdir", metavar="'logdir'", required=True, type=str,
                        help="The directory containing the logs")
    parser.add_argument("-M", "--excludemissingdata", action="store_true",
                        help="Excludes rows with missing data")
    parser.add_argument("-o", "--outputdir", metavar="'outputdir'", required=True, type=str,
                        help="Output directory")
    parser.add_argument("-r", "--replacemissingdatamethod", metavar="'replacemissingdatamethod'",
                        required=False, type=str, default="mean", help="mean|median")
    parser.add_argument("-S", "--scaledata", action="store_true",
                        help="Scale/normalize data")
    parser.add_argument("-s", "--scaledatamethod", metavar="'scaledatamethod'", required=False, type=str,
                        default="RobustScaler", help="MinMaxScaler|Normalizer|RobustScaler|StandardScaler")
    parser.add_argument("-v", "--cvssversion", metavar="'cvssversion'", required=False, type=str,
                        default="2", help="2|3")
    args = parser.parse_args()

    client = Ml.DataPreparation(logDir=args.logdir, outputDir=args.outputdir, cfgPath=args.cfgpath,
                                scaleDataFlag=args.scaledata, scaleDataMethod=args.scaledatamethod,
                                excludeMissingDataFlag=args.excludemissingdata,
                                replaceMissingDataMethod=args.replacemissingdatamethod,
                                cvssVersion=args.cvssversion)
    sys.exit(client.run())
