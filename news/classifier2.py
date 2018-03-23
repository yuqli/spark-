#!urs/bin/python
import re
import pandas
import pickle 

from pyspark import SparkContext

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF,StopWordsRemover,IDF,Tokenizer, Word2Vec
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier 

from pyspark.mllib.evaluation import MulticlassMetrics
# ======================================================================================
#                                       Read Data
# ======================================================================================
if __name__ == "__main__":
    sc = SparkContext(appName="newsclassification")

    sqlContext = SQLContext(sc)
    path = "news/20news-bydate-train/*/*"
    newsGroupRowData=sc.wholeTextFiles(path)
    print("Number of documents read in is:", newsGroupRowData.count())

    # ======================================================================================
    #                                       Clean Data
    # ======================================================================================
    # a function to remove pubctuation
    def removePunctuation(text):
        text = re.sub(r'[\d]', ' ', text)
        text = re.sub(r'[^\w]', ' ', text)
        text = text.lower()
        text = 	" ".join(text.split())
        return text

    newsgroups = newsGroupRowData.map(lambda line: (line[0].split("/")[-1],removePunctuation(line[1]),line[0].split("/")[-2]))

    # convert to a Dataframe
    schemaString = "id text topic"
    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
    schema = StructType(fields)
    df = sqlContext.createDataFrame(newsgroups, schema)

    # Add index
    indexer = StringIndexer(inputCol="topic", outputCol="label")
    indexed = indexer.fit(df).transform(df)
    # take a smaller sample and test!
    # df = indexed.sample(False, 0.1, seed=0).limit(10)
    df = indexed

    # tokenize, remove stopwords
    tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
    remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
    pipeline = Pipeline(stages=[tokenizer,remover])
    clean = pipeline.fit(df)
    df = clean.transform(df)
    train_set, test_set = df.randomSplit([0.9, 0.1], 12345)

    # ======================================================================================
    #                                       Feature Extraction
    # ======================================================================================
    # first use word2Vec
    word2vec = Word2Vec().setVectorSize(100).setSeed(42).setInputCol("filtered").setOutputCol("features")
    model = word2vec.fit(train_set)
    train_set1 = model.transform(train_set)
    test_set1 = model.transform(test_set)

    # now use tf-idf
    hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
    idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(10)
    pipeline = Pipeline(stages=[hashingTF, idf])
    train_set2 = pipeline.fit(train_set).transform(train_set)
    pipeline2 = Pipeline(stages=[hashingTF])
    test_set2 = pipeline2.fit(train_set).transform(test_set)  # use trainset idf to transform test set
    test_set2 = idf.fit(test_set2).transform(test_set2)

    # ======================================================================================
    #                                       Fit Model
    # ======================================================================================
    def fit_nb(train):
        rf = RandomForestClassifier(numTrees=20, maxDepth=20, labelCol="label", seed=42)
        model = rf.fit(train)
        return model

    def get_predictions(model, test):
        result = model.transform(test.select('features'))  # result is a DataFrame
        predictions = result.select('prediction').rdd.map(tuple)
        predictionsAndLabels = predictions.zip(test.select('label', 'id', 'topic').rdd.map(tuple))
        predictionsAndLabels = predictionsAndLabels.map(lambda x: (x[0][0], x[1][0], x[1][1], x[1][2]))  # remove the tuple structure
        predictionsAndLabels = predictionsAndLabels.map(lambda line: Row(prediction = line[0], label = line[1], id=line[2], topic=line[3])).toDF().toPandas()
        return predictionsAndLabels

    model = fit_nb(train_set1)
    pl = get_predictions(model, test_set1)
    pickle.dump(pl, open('pl1.p', 'wb'))
    
    model = fit_nb(train_set2)
    pl = get_predictions(model, test_set2)
    pickle.dump(pl, open('pl2.p', 'wb'))
    sc.stop()
