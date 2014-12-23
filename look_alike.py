from pyspark import SparkContext, SparkConf
from argparse import ArgumentParser
from math import sqrt

APPLICATION_NAME = "Look-Alike Task"
LOG_LINE_FIELDS_NUMBER = 22


def parse_log_line(log_line):
    parts = log_line.split("*")
    if len(parts) < LOG_LINE_FIELDS_NUMBER:
        # invalid log line
        return []
    user_id = parts[3]
    site_id = int(parts[4])
    return [(user_id, site_id)]


def calculate_ratings(data):
    def calculate_site_ratings_for_user(key_value_pair):
        user_id = key_value_pair[0]
        site_ids = key_value_pair[1]

        site_id_to_visits_count = dict()
        max_visits_count = 0.0
        for site_id in site_ids:
            if site_id not in site_id_to_visits_count:
                site_id_to_visits_count[site_id] = 0.0
            site_id_to_visits_count[site_id] += 1
            max_visits_count = max(max_visits_count, site_id_to_visits_count[site_id])

        ratings = dict()
        for site_id, visits_count in site_id_to_visits_count.iteritems():
            site_rating = visits_count / max_visits_count
            ratings[site_id] = site_rating
        return user_id, ratings

    return data.groupByKey().map(calculate_site_ratings_for_user)


def calculate_correlations(ratings_data, target_site_id):
    def collect_rating_pairs(user_ratings_pair):
        ratings = user_ratings_pair[1]
        if target_site_id not in ratings:
            return []
        target_site_rating = ratings[target_site_id]
        return [(site_id, (site_rating, target_site_rating)) for (site_id, site_rating) in ratings.iteritems()
                if site_id != target_site_id]

    def create_combiner(site_ratings_pair):
        rating1 = site_ratings_pair[0]
        rating2 = site_ratings_pair[1]
        return [rating1, rating2, rating1 * rating1, rating2 * rating2, rating1 * rating2, 1]

    def sum_combiner_with_value(combiner, site_ratings_pair):
        rating1 = site_ratings_pair[0]
        rating2 = site_ratings_pair[1]
        combiner[0] += rating1
        combiner[1] += rating2
        combiner[2] += rating1 * rating1
        combiner[3] += rating2 * rating2
        combiner[4] += rating1 * rating2
        combiner[5] += 1
        return combiner

    def sum_combiners(combiner1, combiner2):
        for i in xrange(len(combiner1)):
            combiner1[i] += combiner2[i]
        return combiner1

    def calculate_correlation(key_value_pair):
        site_id = key_value_pair[0]
        combiner_result = key_value_pair[1]
        # obtain combined data
        ratings_sum1 = combiner_result[0]
        ratings_sum2 = combiner_result[1]
        squared_ratings_sum1 = combiner_result[2]
        squared_ratings_sum2 = combiner_result[3]
        ratings_product_sum = combiner_result[4]
        items_number = combiner_result[5]
        # calculate correlation coefficient itself
        mean1 = ratings_sum1 / items_number
        mean2 = ratings_sum2 / items_number
        std_dev1 = sqrt(squared_ratings_sum1 / items_number - mean1 * mean1)
        std_dev2 = sqrt(squared_ratings_sum2 / items_number - mean2 * mean2)
        if std_dev1 == 0 or std_dev2 == 0:
            # correlation will be 0, no need to return value
            return []
        correlation = (ratings_product_sum / items_number - mean1 * mean2) / (std_dev1 * std_dev2)
        return [(site_id, correlation)]

    return ratings_data.flatMap(collect_rating_pairs).\
        combineByKey(create_combiner, sum_combiner_with_value, sum_combiners).\
        flatMap(calculate_correlation)


def predict_user_ratings(site_ratings_data, correlation_map, target_site_id):
    def predict_rating_for_user(user_ratings_pair):
        user_id = user_ratings_pair[0]
        ratings = user_ratings_pair[1]
        # predict ratings only for users who hasn't visited target site
        if target_site_id in ratings:
            return []
        avg_rating = sum(ratings.itervalues(), 0.0) / len(ratings)
        predicted_rating = 0.0
        normalizing_sum = 0.0
        for site_id, site_rating in ratings.iteritems():
            if site_id in correlation_map:
                weight = correlation_map[site_id]
                predicted_rating += (site_rating - avg_rating) * weight
                normalizing_sum += abs(weight)
        if normalizing_sum != 0:
            predicted_rating /= normalizing_sum
        predicted_rating += avg_rating
        return [(user_id, predicted_rating)]

    return site_ratings_data.flatMap(predict_rating_for_user)


def main():
    args_parser = ArgumentParser()
    args_parser.add_argument("--input", help="input files location", required=True)
    args_parser.add_argument("--target-site", help="target site id, for which list of recommended users should be provided", required=True)
    args_parser.add_argument("--users-count", help="number of users in result list", required=True)

    args = args_parser.parse_args()
    input_files_location = args.input
    target_site_id = int(args.target_site)
    users_count = int(args.users_count)

    configuration = SparkConf().setAppName(APPLICATION_NAME)
    spark_context = SparkContext(conf=configuration)
    input_data = spark_context.textFile(input_files_location).flatMap(parse_log_line)
    site_ratings_data = calculate_ratings(input_data).cache()  # cache as this data is used twice
    correlation_map = calculate_correlations(site_ratings_data, target_site_id).collectAsMap()
    predicted_ratings_by_users = predict_user_ratings(site_ratings_data, correlation_map, target_site_id)
    result = predicted_ratings_by_users.takeOrdered(users_count, lambda x: -x[1])

    for item in result:
        print item[0], ":", item[1]

if __name__ == "__main__":
    main()