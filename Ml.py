#!/usr/bin/env python3
import csv
import numpy as np
import pandas as pd
import re
import sys
import time
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

class ModelGeneration(object):

    def __init__(self, outputDir, cfgPath="ml.json", kFolds=2):
        self.outputDir = outputDir
        self.cfgPath = cfgPath
        self.kFolds = int(kFolds)
        self.dfData = None
        self.dfFeatures = None
        self.dfLabel = None
        self.featureNames = None
        self.labelName = None
        self.classifiers = {
            "Decision Tree": tree.DecisionTreeClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
            #"Linear SVM": SVC(),
            #"Logistic Regression": LogisticRegression(),
            "Naive Bayes": GaussianNB(),
            "Nearest Neighbors": KNeighborsClassifier(),
            #"Neural Net": MLPClassifier(alpha = 1),
            "Random Forest": RandomForestClassifier(n_estimators=1000)
        }

    def build(self, inputDataPath):
        Msg.raw("Input: {0}".format(inputDataPath))
        self.ingestData(inputDataPath)
        kf = KFold(n_splits=self.kFolds, shuffle=True, random_state=None)
        X = self.dfFeatures.values.astype(dtype="int64")
        y = self.dfLabel.values.astype(dtype="int64")
        for trainIndex, testIndex in kf.split(X):
            xTrain, xTest = X[trainIndex], X[testIndex]
            yTrain, yTest = y[trainIndex], y[testIndex]
            print(xTrain)
            for name, classifier in sorted(list(self.classifiers.items())):
                startTime = time.clock()
                classifier.fit(xTrain, yTrain)
                endTime = time.clock()
                deltaTime = endTime - startTime
                prediction = classifier.predict(xTest)
                trainScore = classifier.score(xTrain, yTrain)
                testScore = classifier.score(xTest, yTest)
                Msg.raw("Classifier: {0}".format(name))
                Msg.raw("Train: {0}, Test: {1}\n".format(Number.asFloat(trainScore),
                                                         Number.asFloat(testScore)))
            print("")
        return 0

    def ingestData(self, inputDataPath):
        self.dfData = pd.read_csv(inputDataPath)
        self.labelName = self.dfData.columns.values[0].strip()
        self.featureNames = self.dfData.columns.values[3:]
        self.dfLabel = self.dfData[self.labelName]
        self.dfFeatures = self.dfData[self.featureNames]

    def run(self, inputDataPath, isDirectoryFlag=False):
        if isDirectoryFlag:
            for path in Dir.getFiles(inputDataPath, recursiveFlag=True):
                if File.getExtension(path) == "csv":
                    self.build(path)
        else:
            pass
        return 0


class DataPreparation(object):

    def __init__(self, logDir, outputDir, cfgPath="ml.json",
                 scaleDataFlag=False, scaleDataMethod="RobustScaler",
                 excludeMissingDataFlag=False, replaceMissingDataMethod="mean"):
        self.logDir = logDir
        self.outputDir = outputDir
        self.cfgPath = cfgPath
        self.scaleDataFlag = scaleDataFlag
        self.scaleDataMethod = scaleDataMethod
        self.excludeMissingDataFlag = excludeMissingDataFlag
        self.replaceMissingDataMethod = replaceMissingDataMethod
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
            self.dfs[label][labelName] = self.dfs[label][labelName].apply(lambda x: Cvss.getSeverityV2(x))
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

