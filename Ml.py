#!/usr/bin/env python3
import csv
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import xlsxwriter
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from JunkDrawer import Data, Dir, File, Msg, Number

pd.options.display.float_format = '{:,.2f}'.format

class Cvss:

    @staticmethod
    def getSeverityV2(severityValue):
        if severityValue == 0.0:
            return 0
        elif severityValue > 0.0 and severityValue <= 3.9:
            return 1
        elif severityValue > 3.9 and severityValue <= 6.9:
            return 2
        elif severityValue > 6.9 and severityValue <= 10.0:
            return 3
        return None

    @staticmethod
    def getSeverityV3(severityValue):
        if severityValue == 0.0:
            return 0
        elif severityValue > 0.0 and severityValue <= 3.9:
            return 1
        elif severityValue > 3.9 and severityValue <= 6.9:
            return 2
        elif severityValue > 6.9 and severityValue <= 8.9:
            return 3
        elif severityValue > 8.9 and severityValue <= 10.0:
            return 4
        return None


class Ml(object):

    def __init__(self):
        self.classifiers = {
            "Decision Tree": tree.DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=1000),
            "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
            "Linear SVM": SVC(kernel="linear", C=0.025),
            "Naive Bayes": GaussianNB(),
            "Nearest Neighbors": KNeighborsClassifier(3),
            "Neural Net": MLPClassifier(alpha = 1, max_iter=10000),
            "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        }


class ReportBuilder(Ml):

    def __init__(self, inputDataPath, outputDataPath):
        super().__init__()
        self.inputDataPath = inputDataPath
        self.outputDataPath = outputDataPath
        self.df = None

    def run(self):
        self.df = pd.read_csv(self.inputDataPath)
        wb= xlsxwriter.Workbook(self.outputDataPath)
        fmtBold = wb.add_format({'bold': True})
        wsSummary = wb.add_worksheet("Summary")
        wsData = wb.add_worksheet("Data")
        wsSummary.set_column("A:I", 18)
        wsData.set_column("A:I", 18)
        columns = sorted(self.df.columns)
        columns.insert(0, "K")
        i = 0
        for j in range(0, len(columns)):
            wsData.write(i, j, columns[j], fmtBold)
        i = 1
        beginRow = i
        for index, row in self.df.iterrows():
            wsData.write(i, 0, i)
            for j in range(1, len(columns)):
                wsData.write(i, j, row[columns[j]])
                j += 1
            i += 1
        endRow = i
        columns.pop(0)
        stats = ["Classifier", "Mean", "Median", "Std", "Var", "Min", "Max"]
        for j in range(0, len(stats)):
            wsSummary.write(0, j, stats[j], fmtBold)
        for i in range(0, len(columns)):
            classifier = columns[i]
            sMean = Number.asFloat(np.mean(self.df[classifier]), 3)
            sMedian = Number.asFloat(np.median(self.df[classifier]), 3)
            sStd = Number.asFloat(np.std(self.df[classifier]), 3)
            sVar = Number.asFloat(np.var(self.df[classifier]), 3)
            sMin = np.min(self.df[classifier])
            sMax = np.max(self.df[classifier])
            stats = [classifier, sMean, sMedian, sStd, sVar, sMin, sMax]
            for j in range(0, len(stats)):
                if j == 0:
                    wsSummary.write(i+1, j, stats[j], fmtBold)
                else:
                    wsSummary.write(i+1, j, stats[j])
        wb.close()
        Msg.raw(self.df)


class ModelGeneration(Ml):

    def __init__(self, inputDataPath, outputDataPath, kFolds=2,
                 randomizeDataFlag=False, cycles=1):
        super().__init__()
        self.inputDataPath = inputDataPath
        self.outputDataPath = outputDataPath
        self.kFolds = int(kFolds)
        self.randomizeDataFlag = randomizeDataFlag
        self.cycles = cycles
        self.dfData = None
        self.dfFeatures = None
        self.dfLabel = None
        self.featureNames = None
        self.labelName = None

    def build(self, inputDataPath):
        Msg.raw("Analyzing <-> {0}".format(inputDataPath))
        self.ingestData(inputDataPath)
        kf = KFold(n_splits=self.kFolds, shuffle=self.randomizeDataFlag, random_state=None)
        X = self.dfFeatures.values
        y = self.dfLabel.values.astype(dtype="int64")
        scoreKeeper = {}
        for name, classifier in sorted(list(self.classifiers.items())):
            scores = cross_val_score(classifier, X, y, cv=kf)
            if not name in scoreKeeper:
                scoreKeeper[name] = []
            scoreKeeper[name].extend(scores)
        return scoreKeeper

    def ingestData(self, inputDataPath):
        self.dfData = pd.read_csv(inputDataPath)
        self.labelName = self.dfData.columns.values[0].strip()
        self.featureNames = self.dfData.columns.values[3:]
        self.dfLabel = self.dfData[self.labelName]
        self.dfFeatures = self.dfData[self.featureNames]

    def initialize(self):
        Msg.show("Initializing")
        Dir.make(File.getDirectory(self.outputDataPath))

    def run(self):
        self.initialize()
        scoreKeeper = Data.buildDictByKeys(self.classifiers.keys(), [])
        for i in range(0, self.cycles):
            Msg.show("Cycle {0} of {1}".format(i+1, self.cycles))
            scores = self.build(self.inputDataPath)
            for classifier in scores.keys():
                scoreKeeper[classifier].extend(scores[classifier])
        df = pd.DataFrame.from_dict(scoreKeeper)
        df.to_csv(self.outputDataPath, index=False, float_format='%.2f')
        Msg.show("Saved results -> {0}".format(self.outputDataPath))


class DataPreparation(object):

    def __init__(self, logDir, outputDir, cfgPath="ml.json",
                 scaleDataFlag=False, scaleDataMethod="RobustScaler",
                 excludeMissingDataFlag=False, replaceMissingDataMethod="mean",
                 cvssVersion="2"):
        self.logDir = logDir
        self.outputDir = outputDir
        self.cfgPath = cfgPath
        self.scaleDataFlag = scaleDataFlag
        self.scaleDataMethod = scaleDataMethod
        self.excludeMissingDataFlag = excludeMissingDataFlag
        self.replaceMissingDataMethod = replaceMissingDataMethod
        self.cvssVersion = cvssVersion
        self.logPaths = None
        self.features = None
        self.labels = None
        self.aliases = None
        self.labelMap = None
        self.dfs = None
        self.paths = None

    def _sast1(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.labels[name])
        return df

    def _sast2(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.labels[name])
        return df

    def _sonarqube(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.labels[name])
        return df

    def _cloc(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.features[name])
        return df

    def _dependencycheck(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.features[name])
        return df

    def _git(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.features[name])
        return df

    def _lizard(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.features[name])
        return df

    def _retire(self, logPaths):
        name = sys._getframe().f_code.co_name.replace("_", "")
        df = self.ingestLogData(logPaths, name, self.features[name])
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

    def ingestLogData(self, logPaths, name, attributes):
        Msg.show("Ingesting {0} logs".format(name))
        if not "projectName" in attributes:
            attributes.insert(0, "projectName")
        if not "sessionId" in attributes:
            attributes.insert(0, "sessionId")
        samples = []
        for logPath in logPaths:
            results = File.read(path=logPath, asJsonFlag=True)
            samples.append(Data.getDictByKeys(results, attributes, None))
        df = pd.DataFrame(samples)
        if self.excludeMissingDataFlag:
            df.dropna(inplace=True)
        else:
            if self.replaceMissingDataMethod == "median":
                df.fillna(df.median(), inplace=True)
            else:
                df.fillna(df.mean(), inplace=True)
        alias = self.aliases[name]
        return df

    def initialize(self):
        Msg.show("Initializing")
        Msg.show("CVSS scoring version: {0}".format(self.cvssVersion))
        Msg.show("Scale data: {0}".format(self.scaleDataFlag))
        cfg = File.read(path=self.cfgPath, asJsonFlag=True)
        self.features = cfg["features"]
        self.labels = cfg["labels"]
        self.aliases = cfg["aliases"]
        self.labelMap = {}
        for label in self.labels:
            self.labelMap[label] = self.labels[label][0]
        Dir.make(self.outputDir)
        self.paths = []
        if self.excludeMissingDataFlag:
            Msg.show("Excluding rows with missing data")
        else:
            Msg.show("Replacing missing data using '{0}' method".format(
                self.replaceMissingDataMethod))

    def _prepareColumns(self, cols, alias, excludeCols=None):
        newCols = []
        for i in range(0, len(cols)):
            if cols[i] in excludeCols:
                newCols.append(cols[i])
                continue
            newCols.append("{0}-{1}".format(alias, cols[i]))
        return newCols

    def run(self):
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
            labelName = self.labelMap[label]
            if self.cvssVersion == 2:
                self.dfs[label][labelName] = self.dfs[label][labelName].apply(lambda x: Cvss.getSeverityV2(x))
            else:
                self.dfs[label][labelName] = self.dfs[label][labelName].apply(lambda x: Cvss.getSeverityV3(x))
        for feature in self.features:
            self.saveDataFeature(feature, self.dfs[feature])
        self.saveData()
        if self.scaleDataFlag:
            self.scaleData()
        return 0

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
            self.paths.append(path)
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
            self.paths.append(path)
            df.to_csv(path, index=False, float_format='%.2f')

    def scaleData(self):
        scaler = preprocessing.RobustScaler()
        for path in self.paths:
            dfData = pd.read_csv(path)
            labelName = dfData.columns.values[0].strip()
            joinNames = dfData.columns.values[1:3]
            featureNames = dfData.columns.values[3:]
            allNames =  [labelName] + joinNames.tolist() + featureNames.tolist()
            df = pd.DataFrame(columns=allNames)
            df[labelName] = dfData[labelName].values
            df[joinNames] = dfData[joinNames].values
            df[featureNames] = scaler.fit_transform(dfData[featureNames])
            Msg.show("Rescaling {0}".format(path))
            df.to_csv(path, index=False, float_format='%.2f')

