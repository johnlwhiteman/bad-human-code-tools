#!/usr/bin/env python3
import numpy as np
import pandas as pd
import re
import sys
from JunkDrawer import Data, Dir, File, Msg

pd.options.display.float_format = '{:,.2f}'.format

class Ml(object):

    def __init__(self, logDir, outputDir, reportPath, cfgPath="ml.json",
                 excludeMissingDataFlag=False, replaceMissingDataMethod="mean"):
        self.logDir = logDir
        self.outputDir = outputDir
        self.reportPath = reportPath
        self.cfgPath = cfgPath
        self.excludeMissingDataFlag = excludeMissingDataFlag
        self.replaceMissingDataMethod = replaceMissingDataMethod
        self.logPaths = None
        self.features = None
        self.labels = None
        self.aliases = None
        self.dfs = None

    def _cloc(self, logPaths):
        df = self.ingestLogData(logPaths, "cloc", self.features["cloc"])
        return df

    def _dependencycheck(self, logPaths):
        df = self.ingestLogData(logPaths, "dependencycheck", self.features["dependencycheck"])
        return df

    def _git(self, logPaths):
        df = self.ingestLogData(logPaths, "git", self.features["git"])
        return df

    def _lizard(self, logPaths):
        df = self.ingestLogData(logPaths, "lizard", self.features["lizard"])
        return df

    def _retire(self, logPaths):
        df = self.ingestLogData(logPaths, "retire", self.features["retire"])
        return df

    def _sonarqube(self, logPaths):
        df = self.ingestLogData(logPaths, "sonarqube", self.labels["sonarqube"])
        return df

    def ingestLogs(self):
        self.logPaths = {
            "features": Data.buildDictByKeys(self.features, []),
            "labels": Data.buildDictByKeys(self.labels, [])
        }
        for logPath in Dir.getFiles(path=self.logDir, recursiveFlag=True, excludeDirFlag=False):
            if not re.search("json", File.getExtension(logPath), re.IGNORECASE):
                continue
            if re.search("_raw.json", logPath, re.IGNORECASE):
                continue
            tokens = File.getNameOnly(logPath).split("_")
            context = tokens[0].strip().lower()
            sessionId = tokens[-1]
            if context in self.features:
                self.logPaths["features"][context].append(logPath)
            elif context in self.labels:
                self.logPaths["labels"][context].append(logPath)
            else:
                continue
        for featureOrLabel in self.logPaths.keys():
            for context in self.logPaths[featureOrLabel].keys():
                if len(self.logPaths[featureOrLabel][context]) < 1:
                    self.logPaths[featureOrLabel][context] = None
        return self.logPaths

    def ingestLogData(self, logPaths, featureName, features):
        Msg.show("Ingesting {0} logs".format(featureName))
        if not "projectName" in features:
            features.insert(0, "projectName")
        if not "sessionId" in features:
            features.insert(0, "sessionId")
        samples = []
        for logPath in logPaths:
            results = File.read(path=logPath, asJsonFlag=True)
            samples.append(Data.getDictByKeys(results, features, None))
        df = pd.DataFrame(samples)
        if self.excludeMissingDataFlag:
            df.dropna(inplace=True)
        else:
            if self.replaceMissingDataMethod == "median":
                df.fillna(df.median(), inplace=True)
            else:
                df.fillna(df.mean(), inplace=True)
        alias = self.aliases[featureName]
        return df

    def initialize(self):
        cfg = File.read(path=self.cfgPath, asJsonFlag=True)
        self.features = cfg["features"]
        self.labels = cfg["labels"]
        self.aliases = cfg["aliases"]
        Dir.make(self.outputDir)

    def _prepareColumns(self, cols, alias, excludeCols=None):
        newCols = []
        for i in range(0, len(cols)):
            if cols[i] in excludeCols:
                newCols.append(cols[i])
                continue
            newCols.append("{0}-{1}".format(alias, cols[i]))
        return newCols

    def prepareData(self):
        self.initialize()
        self.dfs = {}
        if self.logPaths is None:
            self.ingestLogs()
        for context in list(self.features.keys()) + list(self.labels.keys()):
            methodName = "_{0}".format(context)
            if not methodName in dir(self):
                Msg.abort("Missing expected class method handler for {0}".format(context))
        for feature in self.features:
            self.dfs[feature] = \
                getattr(self, "_{0}".format(feature))(self.logPaths["features"][feature])
        for label in self.labels:
            self.dfs[label] = \
                getattr(self, "_{0}".format(label))(self.logPaths["labels"][label])
        for feature in self.features:
            self.saveDataFeature(feature, self.dfs[feature])
        self.saveData()

    def saveData(self):
        joinCols = ["sessionId", "projectName"]
        joinColsCnt = len(joinCols)
        for labelName in self.labels:
            path = "{0}/{1}.csv".format(self.outputDir, labelName)
            df = None
            labelDf = self.dfs[labelName].copy(deep=True)
            labelCols = self._prepareColumns(
                            labelDf.columns.values.tolist(),
                            self.aliases[labelName],
                            joinCols)
            labelDf.columns = labelCols
            for featureName in self.features:
                featureDf = self.dfs[featureName].copy(deep=True)
                featureCols = self._prepareColumns(
                                featureDf.columns.values.tolist(),
                                self.aliases[featureName],
                                joinCols)
                featureDf.columns = featureCols
                if df is None:
                    df = labelDf.merge(featureDf, how="inner", on=joinCols)
                else:
                    df = df.merge(featureDf, how="inner", on=joinCols)
            Msg.show("Saving {0} -> {1}".format(labelName, path))
            df.to_csv(path, index=False, float_format='%.2f')

    def saveDataFeature(self, featureName, featureDf):
        joinCols = ["sessionId", "projectName"]
        featureCols = self._prepareColumns(
                        featureDf.columns.values.tolist(),
                        self.aliases[featureName],
                        joinCols)
        featureDf.columns = featureCols
        for labelName in self.labels:
            path = "{0}/{1}-{2}.csv".format(self.outputDir, labelName, featureName)
            labelDf = self.dfs[labelName].copy(deep=True)
            labelCols = self._prepareColumns(
                            labelDf.columns.values.tolist(),
                            self.aliases[labelName],
                            joinCols)
            labelDf.columns = labelCols
            df = labelDf.merge(featureDf, how="inner", on=joinCols)
            Msg.show("Saving {0}/{1} -> {2}".format(labelName, featureName, path))
            df.to_csv(path, index=False, float_format='%.2f')

