#!usr/bib/python

from pyspark import SparkContext

def parseFriends(line):
    """Parses a line containing users and friends into [user, (friends)]."""
    #parts = re.split('\t', line)
    parts = line.split('\t')
    #parts[1] = re.split(',', parts[1])
    parts[1] = set(parts[1].split(','))
    return parts[0], parts[1]

# these functions are to be used with combineByKey later
# they serve to combine the tuple values to a list of tuples
def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a

# this function is to sort a list of tuples by the second element, descending order
def sort_list_of_tuples(l):
    return sorted(l, key=lambda tup: -tup[1])

NO_of_Recommendations = 10

if __name__ == "__main__":
    sc =SparkContext()
    lines = sc.textFile('soc-LiveJournal1Adj.txt')
    # parse friends. Now item = [userID, friend_list]
    user_friends = lines.map(parseFriends).cache()
    # user_friends = user_friends.take(10)
    # user_friends = sc.parallelize(user_friends)
    # cartesian with itself. Now individual item = [ (userID, [friend_list]),(userID, [friend_list])
    user_friends = user_friends.cartesian(user_friends)
    # remove items that are have the same user in it twice
    user_friends = user_friends.filter(lambda x: x[0][0] != x[1][0])
    # remove user that are in each other's friend list
    user_friends = user_friends.filter(lambda x: x[1][0] not in x[0][1])
    # calculate the number of common friends. Now item = [ userID, (another user ID, numbber of common_friends)]
    #user_friends = user_friends.map(lambda x: (x[0][0], (x[1][0], len(set(x[0][1]) - (set(x[0][1]) - set(x[1][1]))))))
    user_friends = user_friends.map(lambda x: (x[0][0], (x[1][0], len(x[0][1].intersection(x[1][1])))))
    user_friends = user_friends.combineByKey(to_list, append, extend)
    user_friends = user_friends.map(lambda x: (x[0], sort_list_of_tuples(x[1])))
    user_friends = user_friends.map(lambda x: (x[0], x[1][0: NO_of_Recommendations]))
    user_friends = user_friends.flatMap(lambda x:[(x[0], item[0]) for item in x[1]])
    # user_friends.toDF().toPandas().to_csv("result.csv",sep="\t",header=False,index=False)
    user_friends.saveAsTextFile("results2")
