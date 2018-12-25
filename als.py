#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is an example implementation of ALS for learning how to use Spark. Please refer to
pyspark.ml.recommendation.ALS for more conventional use.

This example requires numpy (http://www.numpy.org/)
"""
from __future__ import print_function

import sys

import numpy as np
from numpy import matrix
from pyspark.sql import SparkSession
import csv

LAMBDA = 0.01   # regularization
np.random.seed(42)


def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))


def update(i, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * np.asmatrix(ratings[i, :]).T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)


if __name__ == "__main__":

    """
    Usage: als [M] [U] [F] [iterations] [partitions]"
    """

    print("""WARN: This is a naive implementation of ALS and is given as an
      example. Please use pyspark.ml.recommendation.ALS for more
      conventional use.""", file=sys.stderr)

    spark = SparkSession\
        .builder\
        .appName("PythonALS")\
        .getOrCreate()

    sc = spark.sparkContext
    result = []
    filename = sys.argv[1] if len(sys.argv) > 1 else "ratings_task2.csv"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 671
    U = int(sys.argv[3]) if len(sys.argv) > 3 else 9066
    F = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    ITERATIONS = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    partitions = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    outputfile = sys.argv[7] if len(sys.argv) > 7 else "out_task2.txt"
    ms = np.asmatrix(np.ones((M, F)))
    us = np.asmatrix(np.ones((U, F)))
    indexU = 0
    dict = {}
    k = 0
    R = np.zeros((M,U))
    with open(filename, "rb") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        next(spamreader, None)
        list = []
        for row in spamreader:
            # if indexU == 0:
            #     print (row[0])
            value = row[0].split(",")[:-1]
            # print (value)
            if value[1] not in dict:
                dict[value[1]] = indexU
                indexU += 1
            xx = int(value[0]) - 1
            yy = dict[value[1]]
            R[xx, yy] = float(value[2])
            k += 1
    Rb = sc.broadcast(np.asmatrix(R))
    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, usb.value, Rb.value)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, msb.value, Rb.value.T)) \
               .collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        result.append(error)
with open("./" + outputfile,"wb") as file:
    for item in result:
        file.write("%.4f\n" % item)
    spark.stop()
