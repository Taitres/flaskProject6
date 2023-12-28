import json
import math
import random
import time
from flask import Flask, render_template
from flask import jsonify, request, Response
from collections import Counter
import redis
from flask import Flask, request, jsonify, render_template
import math
import time
import random
from nltk.internals import Counter
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from pydocumentdb import document_client

# redis_client = redis.StrictRedis(host='r-bp1t5jikzfiac5go4lpd.redis.rds.aliyuncs.com',password="wasd8456@", port=6379, db=0)
app = Flask(__name__)
redis_client = redis.StrictRedis(host='r-bp1t5jikzfiac5go4lpd.redis.rds.aliyuncs.com',password="wasd8456@", port=6379, db=0)
from pydocumentdb import document_client

# Azure Cosmos DB 连接信息
ENDPOINT = "https://tutorial-uta-cse6332.documents.azure.com:443/"
MASTERKEY = "fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw=="
DATABASE_ID = "tutorial"
COLLECTION_ID1 = "us_cities"
COLLECTION_ID2 = "reviews"

# 连接到 Azure Cosmos DB
client = document_client.DocumentClient(ENDPOINT, {'masterKey': MASTERKEY})

# 查询城市数据
def get_cities_data():
    query1 = "SELECT c.city, c.lat, c.lng c.population FROM c"
    options = {"enableCrossPartitionQuery": True}  # 如果集合是分区集合，需要启用跨分区查询

    # 执行查询
    cities_data_q = list(client.QueryDocuments(f"dbs/{DATABASE_ID}/colls/{COLLECTION_ID1}", query1, options))

    cities_data = []
    for item in cities_data_q:
        cities_data.append({
            "city": item['city'],
            "lat": item['lat'],
            "lng": item['lng'],
            "population": item['population']
        })

    return cities_data


def get_cities():
    sql = "SELECT c.city, c.lat, c.lng, c.population FROM c"
    o = {"enableCrossPartitionQuery": True}
    r = list(client.QueryDocuments(f"dbs/{DATABASE_ID}/colls/{COLLECTION_ID1}", sql, o))
    c = []
    for i in r:
        c.append({
            "city": i['city'],
            "lat": i['lat'],
            "lng": i['lng'],
            "population": i["population"]
        })
    return c

def get_reviews_data():
    query2 = "SELECT c.city, c.score FROM c"
    options = {"enableCrossPartitionQuery": True}  # 如果集合是分区集合，需要启用跨分区查询

    # 执行查询

    reviews_data_q = list(client.QueryDocuments(f"dbs/{DATABASE_ID}/colls/{COLLECTION_ID2}", query2, options))


    rev = []
    for item in reviews_data_q:
        rev.append({
            "city": item['city'],
            "score": item['score']
        })

    return rev
def get_reviews():
    query2 = "SELECT TOP 100 c.city, c.review FROM c"
    options = {"enableCrossPartitionQuery": True}  # 如果集合是分区集合，需要启用跨分区查询

    # 执行查询

    reviews_data_q = list(client.QueryDocuments(f"dbs/{DATABASE_ID}/colls/{COLLECTION_ID2}", query2, options))


    reviews_data = []
    for item in reviews_data_q:
        reviews_data.append({
            "city": item['city'],
            "review": item['review']
        })

    return reviews_data

@app.route('/', methods=['GET'])
def hello():  # put application's code here
    return render_template('b.html')

@app.route('/2', methods=['GET'])
def world():  # put application's code here
    return render_template('a.html')

@app.route('/3', methods=['GET'])
def world2():  # put application's code here
    return render_template('c.html')

@app.route('/closest_cities', methods=['GET'])
def closest_cities():
    city_name = request.args.get('city')
    page_size = 50
    page = int(request.args.get('page'))

    # 尝试从 Redis 中获取缓存数据
    start_redis_time = time.time()
    cached_result = redis_client.get(f'closest_cities:{city_name}:{page}')
    end_redis_time = time.time()
    if cached_result:
        # 如果缓存数据存在，直接返回
        print("从缓存中获取数据")
        redis_time = int((end_redis_time - start_redis_time) * 1000)  # 转换为毫秒
        return Response(cached_result, content_type='application/json')

    # 如果缓存数据不存在，执行计算和排序操作
    cities_data = get_cities()
    start_time = time.time()

    # Fetch data from Cosmos DB (replace this with actual Cosmos DB query)
    city_data = next((city for city in cities_data if city["city"] == city_name), None)

    if not city_data:
        return jsonify({"error": "City not found"}), 404

    # Process data and calculate Eular distances
    all_cities_distances = []
    for other_city in cities_data:
        if other_city["city"] != city_name:
            distance = calculate_eular_distance(city_data["lat"], city_data["lng"], other_city["lat"], other_city["lng"])
            all_cities_distances.append({"city": other_city["city"], "distance": distance})

    # Sort cities by distance
    sorted_cities = sorted(all_cities_distances, key=lambda x: x["distance"])

    # Paginate the result
    start_index = page * page_size
    end_index = (page + 1) * page_size
    if (end_index > len(sorted_cities)):
        end_index = len(sorted_cities)
    print(len(sorted_cities))
    paginated_result = sorted_cities[start_index:end_index]

    end_time = time.time()
    computing_time = int((end_time - start_time) * 1000)  # 转换为毫秒
    # Convert result to JSON format
    result_json = json.dumps({"result": paginated_result, "time_of_computing": computing_time})

    # 将结果缓存到 Redis 中，设置过期时间（假设设置为 1 小时）
    redis_client.setex(f'closest_cities:{city_name}:{page}', 3600, result_json)

    # Return response
    return Response(result_json, content_type='application/json')
def calculate_eular_distance(x1, y1, x2, y2):
    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@app.route('/average_review', methods=['GET'])
def average_review():
    city_name = request.args.get('city')
    page_size = 10
    page = int(request.args.get('page', 0))

    # 尝试从 Redis 中获取缓存数据
    cached_result = redis_client.get(f'average_review:{city_name}:{page}')

    if cached_result:
        # 如果缓存数据存在，直接返回
        print("从缓存中获取数据")
        return Response(cached_result, content_type='application/json')

    cities_data = get_cities_data()
    city_data = next((city for city in cities_data if city["city"] == city_name), None)

    if not city_data:
        return jsonify({"error": "City not found"}), 404

    # Process data and calculate Eular distances
    all_cities_distances = []
    for other_city in cities_data:
        if other_city["city"] != city_name:
            distance = calculate_eular_distance(city_data["lat"], city_data["lng"], other_city["lat"], other_city["lng"])
            all_cities_distances.append({"city": other_city["city"], "distance": distance})

    # Sort cities by distance
    sorted_cities = sorted(all_cities_distances, key=lambda x: x["distance"])

    display_cities = [city["city"] for city in sorted_cities]

    reviews_data = get_reviews_data()

    result = []
    for city1 in display_cities:
        # Fetch reviews data from Cosmos DB for the city
        reviews_city = [review for review in reviews_data if review["city"] == city1]

        if len(reviews_city) != 0 :
            # Calculate the average review score
            total_score = sum(float(review["score"]) for review in reviews_city)
            average_score = total_score / len(reviews_city)
            result.append({"city": city1, "average_review_score": average_score})
    start_index = page * page_size
    end_index = (page + 1) * page_size
    result = result[start_index:end_index]

    print(result)
    result_json = json.dumps({"result": result})
    redis_client.setex(f'average_review:{city_name}:{page}', 3600, result_json)

    return Response(result_json, content_type='application/json')

with open("static/stopwords.txt", "r", encoding="utf-8") as stopwords_file:
    stopwords = set(stopwords_file.read().splitlines())



def calculate_euclidean_distance(city1, city2):
    lat1, lon1 = city1['lat'], city1['lng']
    lat2, lon2 = city2['lat'], city2['lng']
    return math.sqrt((float(lat1) - float(lat2)) ** 2 + (float(lon1) - float(lon2)) ** 2)


# KNN算法实现

def knn_clustering(classes, k, words):
    # 初始化聚类
    clusters = {i: [] for i in range(classes)}

    cities_data = get_cities()

    # 为每个城市分配一个初始类别
    for city in cities_data:
        assigned_class = random.randint(0, classes - 1)
        clusters[assigned_class].append(city)

    # 迭代分配城市到最近的类别
    for city in cities_data:
        distances = []
        for class_id in clusters:
            center = clusters[class_id][0]  # 假设每个类别的第一个城市是中心
            distance = calculate_euclidean_distance(city, center)
            distances.append((class_id, distance))
        distances.sort(key=lambda x: x[1])
        nearest_classes = [class_id for class_id, _ in distances[:k]]
        most_common_class = Counter(nearest_classes).most_common(1)[0][0]
        clusters[most_common_class].append(city)

    return clusters


@app.route('/knn_reviews', methods=['GET'])
def knn_reviews():
    start_time = time.time()

    # 从请求中获取参数
    classes = int(request.args.get('classes', 6))
    k = int(request.args.get('k', 3))
    words = int(request.args.get('words', 100))

    # 执行KNN聚类
    clusters = knn_clustering(classes, k, words)

    # 计算每个类的总人口
    cluster_populations = {class_id: sum(int(city['population']) for city in cities) for class_id, cities in
                           clusters.items()}

    # 计算响应时间
    elapsed_time = (time.time() - start_time) * 1000

    response = {
        'clusters': [{'classId': class_id, 'population': population} for class_id, population in
                     cluster_populations.items()],
        'time_ms': elapsed_time
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
